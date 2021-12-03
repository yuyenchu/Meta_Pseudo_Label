# Meta Pseudo Label

## Description

Recreating the Meta Pseudo Label(MPL), a State-of-the-Art  semi-supervised learning method proposed by the Google team.

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

## Help

Email us for more detail

## Authors

Andrew Yu - [Github](https://github.com/yuyenchu) - andrew7011616@gmail.com

## Version History

* 0.1-beta
    * basic functions

## Todo

- [ ] add save best version of model/checkpoint
- [ ] add grid search for parameter
- [ ] make abstract class for future implementation
- [ ] debugging

## License

This project is licensed under the GNU License - see the LICENSE.md file for details

## Acknowledgments

Inspiration and snippet from
* [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580)
* [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)