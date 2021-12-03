from datetime import datetime
import os
from os.path import join, isdir
import glob
import numpy as np
import sys
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses, optimizers, metrics, models, Input
from typing import Tuple
import model

DEBUG = False

class UDA:
    def __init__(self, augmentation_methods, out_shape=(28, 28, 3), seed=None, distort_fn=lambda x:x):
        self.aug_map = {
            'brightness':   tf.image.random_brightness,
            'contrast':     tf.image.random_contrast,
            'crop':         tf.image.random_crop,
            'flip_lr':      tf.image.random_flip_left_right,
            'flip_ud':      tf.image.random_flip_up_down,
            'hue':          tf.image.random_hue,
            'quality':      tf.image.random_jpeg_quality,
            'rotate':       tf.keras.preprocessing.image.random_rotation,
            'saturation':   tf.image.random_saturation,
            'distortion':   self.distort_wrap(distort_fn)
        }
        self.augs = list(augmentation_methods)
        self.out_shape = out_shape
        if (seed is not None and seed > 0 and type(seed)==int):
            self.seed = seed
        else:
            self.seed = None
        
    def distort_wrap(self, func):
        vfunc = np.vectorize(func)
        def distort(img):
            it = np.nditer(img, flags=['multi_index'])
            res = np.zeros(img.shape, dtype=img.dtype)
            for x in it:
                res[it.multi_index] = vfunc(img[it.multi_index], *it.multi_index)
            return res
        return distort

    def predict(self, model, data):
        # return label index, not one hot
        out = data
        for aug, args in self.augs:
            out = self.aug_map[aug](out, seed=self.seed, **args)
        out = tf.image.resize(out, self.out_shape[:-1])
        return tf.argmax(model(out), axis=1)
    
    def aug(self, data):
        # return label index, not one hot
        out = data
        for aug, args in self.augs:
            out = self.aug_map[aug](out, seed=self.seed, **args)
        return tf.image.resize(out, self.out_shape[:-1])

class DataGenerator:
    def __init__(self, path, classes=None, out_shape=(28,28,3), buffer_size: int =512, batch_size: int =128, unlabel_batch_size: Tuple[int, None] =None):
        self.path = path
        self.out_shape  = out_shape

        if (classes is not None):
            self.classes = list(classes)
        else:
            self.classes = []
            for f in os.listdir(join(self.path,'labeled')):
                if isdir(join(self.path, 'labeled', f)):
                    self.classes.append(f)
        self.classes    = np.array(self.classes)
        print(f'classes: {self.classes}')

        if (unlabel_batch_size is None):
            unlabel_batch_size = batch_size*8
            
        self.batch_size = batch_size
        self.unlabel_batch_size = unlabel_batch_size
        self.elements = len(glob.glob(join(self.path,'unlabeled/*jpg')))
        self.steps = self.elements//(unlabel_batch_size)+(0 if self.elements%(unlabel_batch_size)==0 else 1)

        self.labeled  = iter( 
                            tf.data.Dataset.list_files(
                                join(self.path, 'labeled/*/*.jpg')
                            ).map(
                                self.load_image_label,
                                num_parallel_calls=tf.data.AUTOTUNE
                            ).shuffle(
                                buffer_size, 
                                reshuffle_each_iteration=True
                            ).batch(
                                self.batch_size
                            ).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        self.unlabeled  = iter(
                            tf.data.Dataset.list_files(           
                                join(self.path, 'unlabeled/*.jpg')
                            ).map(
                                self.load_image_unlabel,
                                num_parallel_calls=tf.data.AUTOTUNE
                            ).shuffle(
                                buffer_size, 
                                reshuffle_each_iteration=True
                            ).batch(
                                self.unlabel_batch_size
                            ).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        self.test   =   tf.data.Dataset.list_files(
                            join(self.path, 'testing/*/*.jpg')
                        ).map(
                            self.load_image_label,
                            num_parallel_calls=tf.data.AUTOTUNE
                        ).shuffle(
                            buffer_size, 
                            reshuffle_each_iteration=True
                        ).batch(batch_size)
    @tf.function
    def get_label(self, file_path):
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == self.classes
        # Integer encode the label
        return tf.argmax(one_hot)

    @tf.function
    def decode_img(self, img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=self.out_shape[-1])
        # Resize the image to the desired size
        return tf.image.resize(img, self.out_shape[:-1])

    @tf.function
    def load_image_label(self, file_path):
        label = self.get_label(file_path)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = (self.decode_img(img)-128.0)/128.0
        return img, label

    @tf.function
    def load_image_unlabel(self, file_path):
        img = tf.io.read_file(file_path)
        img = (self.decode_img(img)-128.0)/128.0
        return img
    
    def next_labeled(self):
        return self.labeled.get_next()

    def next_unlabeled(self):
        return self.unlabeled.get_next()

class MPL:
    def __init__(self, data_path, save_path='', verbose=False, s_opt={}, t_opt={}, model=model.get_model,
                data_args={}, uda_args={'augmentation_methods':[('brightness',{'max_delta':0.2})]}):
        self.data = DataGenerator(data_path, **data_args)
        self.UDA = UDA(**uda_args)
        self.student = model(classes=len(self.data.classes))
        self.teacher = model(classes=len(self.data.classes))
        self.loss_fn = losses.SparseCategoricalCrossentropy(reduction=losses.Reduction.NONE)
        self.loss_fn2 = losses.KLDivergence(reduction=losses.Reduction.NONE)
        self.cos_dist = losses.CosineSimilarity(axis=0)
        self.s_optimizer = optimizers.SGD(**s_opt)
        self.t_optimizer = optimizers.SGD(**t_opt)
        self.learning_rate = self.s_optimizer.learning_rate
        self.metrics = {
            'loss/student/unlabel':     metrics.Mean(name='student_ublabeled_loss'),
            'loss/student/label':       metrics.Mean(name='student_labeled_loss'),
            'loss/teacher/unlabel':     metrics.Mean(name='teacher_ublabeled_loss'),
            'loss/teacher/label':       metrics.Mean(name='teacher_labeled_loss'),
            'loss/teacher/uda':         metrics.Mean(name='teacher_uda_loss'),
            'accu/student/label':       metrics.SparseCategoricalAccuracy(name='student_labeled_accuracy'),
            'accu/student/test':        metrics.SparseCategoricalAccuracy(name='student_testing_accuracy'),
            'accu/teacher/label':       metrics.SparseCategoricalAccuracy(name='teacher_labeled_accuracy'),
            'accu/teacher/test':        metrics.SparseCategoricalAccuracy(name='teacher_testing_accuracy')
        }

        if len(save_path)==0:
            save_path = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_path = save_path
        self.ckpt_path = join(self.save_path, 'ckpt')
        self.model_path = join(self.save_path, 'model')

        self.summary_writer = tf.summary.create_file_writer(
            join(self.save_path, 'log')
        )
        self.s_checkpoint = tf.train.Checkpoint(optimizer=self.s_optimizer, model=self.student)
        self.t_checkpoint = tf.train.Checkpoint(optimizer=self.t_optimizer, model=self.teacher)
        
        self.verbose = verbose
        if self.verbose:
            print('-'*24,'MPL Information','-'*24)
            print('Data path:', data_path)
            print('Data generator arguments:')
            pretty(data_args, 1)
            print('UDA arguments:')
            pretty(uda_args, 1)
            print('Model summary:')
            self.student.summary()
        self.evaluate = self._evaluate.get_concrete_function()
        self.train_step = self._train_step.get_concrete_function()
        self.reset_metric = self._reset_metric.get_concrete_function()

    @tf.function
    def _train_step(self):
        print("[0]retracing...",end="")
        data, label    = self.data.next_labeled()
        unlabeled_data = self.data.next_unlabeled()

        label_batch_size   = tf.shape(data)[0]
        unlabel_batch_size = tf.shape(unlabeled_data)[0]

        with tf.GradientTape() as ttape:
            teacher_pred = self.teacher(tf.concat([unlabeled_data, data, self.UDA.aug(unlabeled_data)], 0))
            teacher_pred_ul, teacher_pred_l, teacher_pred_uda = tf.split(teacher_pred, [unlabel_batch_size, label_batch_size, unlabel_batch_size])
        pseudo_label = tf.argmax(tf.stop_gradient(teacher_pred_ul), axis=1)

        with tf.GradientTape() as stape:
            student_pred_ul = self.student(unlabeled_data)
            student_ul_loss = tf.reduce_mean(self.loss_fn(pseudo_label, student_pred_ul))

        student_ul_grad = stape.gradient(student_ul_loss, self.student.trainable_variables)
        self.s_optimizer.apply_gradients(zip(student_ul_grad, self.student.trainable_variables))

        with tf.GradientTape() as nstape:
            student_pred_l  = self.student(data)
            student_l_loss  = self.loss_fn(label, student_pred_l)
        
        student_l_grad  = nstape.gradient(student_l_loss, self.student.trainable_variables)

        g1 = tf.concat([tf.reshape(x, [-1]) for x in student_l_grad], 0)
        g2 = tf.concat([tf.reshape(x, [-1]) for x in student_ul_grad], 0)
        h = tf.stop_gradient(self.learning_rate*self.cos_dist(g1, g2))
        h = tf.where(tf.math.is_nan(h), tf.zeros_like(h), h)
        
        with ttape:
            teacher_ul_loss  = tf.reduce_mean(self.loss_fn(pseudo_label, teacher_pred_ul))
            teacher_l_loss   = tf.reduce_mean(self.loss_fn(label, teacher_pred_l))
            teacher_UDA_loss = tf.reduce_mean(self.loss_fn2(teacher_pred_uda, tf.stop_gradient(teacher_pred_ul)))

            teacher_loss     = h*teacher_ul_loss+teacher_l_loss+teacher_UDA_loss

        teacher_grad = ttape.gradient(teacher_loss, self.teacher.trainable_variables)
        self.t_optimizer.apply_gradients(zip(teacher_grad, self.teacher.trainable_variables))

        self.metrics['accu/student/label'].update_state(label, student_pred_l)
        self.metrics['accu/teacher/label'].update_state(label, teacher_pred_l)
        self.metrics['loss/student/unlabel'].update_state(student_ul_loss, unlabel_batch_size)
        self.metrics['loss/student/label'].update_state(student_l_loss, label_batch_size)
        self.metrics['loss/teacher/unlabel'].update_state(teacher_ul_loss, unlabel_batch_size)
        self.metrics['loss/teacher/label'].update_state(teacher_l_loss, label_batch_size)
        self.metrics['loss/teacher/uda'].update_state(teacher_UDA_loss, label_batch_size)
        
        with self.summary_writer.as_default():
            for name, metric in self.metrics.items():
                tf.summary.scalar(name, metric.result(), step=self.s_optimizer.iterations)
        # tf.print('',label[0:5], ':',tf.argmax(teacher_pred_l[0:5],axis=1),':',tf.argmax(student_pred_l[0:5],axis=1))
        print("done")
        return {
            'loss/student': student_ul_loss, 
            'loss/teacher': teacher_loss,
            'h': h
        }

    @tf.function
    def _evaluate(self):
        print("[1]retracing...",end="")
        tf.print('-'*23,'Evaluation Start','-'*24)
        for x,y in self.data.test:
            self.metrics['accu/student/test'].update_state(y, self.student(x, training=False))
            self.metrics['accu/teacher/test'].update_state(y, self.teacher(x, training=False))
        tf.print('Student accuracy:',self.metrics['accu/student/test'].result())
        tf.print('Teacher accuracy:',self.metrics['accu/teacher/test'].result())
        tf.print('-'*23,'Evaluation End','-'*26,'\n')
        print("done")
        return self.metrics['accu/student/test'].result(), self.metrics['accu/teacher/test'].result()

    @tf.function
    def _reset_metric(self):
        print("[2]retracing...",end="")
        for metric in self.metrics.values():
            metric.reset_state()
        print("done")

    def fit(self, n_epochs):
        padding = len(str(self.data.steps))
        tf.print('\n'+'-'*24,'Training Start','-'*25)
        for epoch in range(1, n_epochs+1):
            tf.print(f'Epoch {epoch: >{len(str(n_epochs))}}/{n_epochs}')
            start = time.time()
            for step in range(self.data.steps):
                values = self.train_step()
            end = time.time()-start
            if DEBUG:
                out = [f'h={tf.strings.as_string(values["h"])}']
                for name, metric in self.metrics.items():
                    out.append(f'{name}: {tf.strings.as_string(metric.result(),3)}')
                tf.print(*out, f' time={end:.3f}s')
            else:
                out = f'student_loss={tf.strings.as_string(self.metrics["loss/student/unlabel"].result(),4)} teacher_loss={tf.strings.as_string(self.metrics["loss/teacher/label"].result(),4)}'
                tf.print(out, f' time={end:.3f}s')
            self.reset_metric()
        tf.print('-'*24,'Training End','-'*27,'\n')
        return self.evaluate()

    def save(self):
        self.student.save(join(self.model_path, 'student.h5'))
        self.teacher.save(join(self.model_path, 'teacher.h5'))
    
    def restore(self):
        self.student = models.load_model(join(self.model_path, 'student.h5'))
        self.teacher = models.load_model(join(self.model_path, 'teacher.h5'))
        self.s_checkpoint.restore(join(self.ckpt_path, 'student'))
        self.t_checkpoint.restore(join(self.ckpt_path, 'teacher'))

    @tf.function
    def debug(self):
        print("[DEBUG]retracing...",end="")
        data, label    = self.data.next_labeled()
        with tf.GradientTape() as ttape:
            pred = self.student(data)
            loss = self.loss_fn(label, pred)
        grad = ttape.gradient(loss, self.student.trainable_weights)
        self.s_optimizer.apply_gradients(zip(grad, self.student.trainable_weights))
        self.metrics['loss/student/label'].update_state(loss)
        self.metrics['accu/student/label'].update_state(label, pred)
        # tf.print('',label[0:5], ':',tf.argmax(pred[0:5],axis=1))
        print("done")
        return {'h':tf.constant(0.0)}

@tf.function
def tf_round(x, decimals = 2):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier

def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent, end=f'{str(key)+" :": <20}')
        if isinstance(value, dict):
            print()
            pretty(value, indent+1)
        else:
            print(f'\t{value}')
        
# def display_progress(step, time, loss, total=1000, c = 100):
#     s = step % total
#     tf.print('\r', end='')
#     tf.print(
#         f'{step:6d}['+'='*((s)//c)+('='if (s+1)%total==0 else '>')+' '*((total-s-1)//c)+']', 
#         end=f' Time: {time:3.2f}s, teacher loss: {loss/(s if s>0 else 1):.5f}, student loss: {loss:.5f}', 
#         flush=True
#     )