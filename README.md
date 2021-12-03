# Meta Pseudo Label

## Description

Recreating the Meta Pseudo Label(MPL), a state of the art semi-supervised learning method proposed by the Google team.  
"Meta Pseudo Labels has a teacher network to generate pseudo labels on unlabeled data to teach a student network. However, unlike Pseudo Labels where the teacher is fixed, the teacher in Meta Pseudo Labels is constantly adapted by the feedback of the student's performance on the labeled dataset. As a result, the teacher generates better pseudo labels to teach the student."

## Getting Started

### Dependencies

* Developed on macOS Mojave with Python 3.8.8

### Installing

* Clone or download the repository
* Install required modules with
```
pip install -r requirements.txt
```
* Edit config.py for your need

### Executing program

* Run with mnist dataset
```
python main.py -p mnist/data -e 15 -c mnist
```

* Using Tensorboard
```
tensorboard --logdir save/log/
```

## Help

* Print help information
```
python main.py -h
```
* Email me for more detail

## Author

Andrew Yu - [Github](https://github.com/yuyenchu) - andrew7011616@gmail.com

## Version History

* 0.1-beta
    * basic functions

## Todo

- [ ] add save best version of model/checkpoint
- [ ] add grid search for parameter
- [ ] add mirrored strategy (distributed training)
- [ ] make abstract class for future implementation
- [ ] debugging

## License

This project is licensed under the GNU License - see the LICENSE.md file for details

## Acknowledgments

Inspiration and snippet from
* [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580)
* [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)