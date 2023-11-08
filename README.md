# Tackling cold-start with deep personalized transfer of user preferences for cross-domain recommendation
This repository is the official implementation of "Tackling cold-start with deep personalized transfer of user preferences for cross-domain recommendation," a paper accepted for publication in the International Journal of Data Science and Analytics. 

Access the full paper: [DOI: 10.1007/s41060-023-00467-9](https://doi.org/10.1007/s41060-023-00467-9)

## Abstract
The recommendation system plays an integral role in our daily lives, from movies to medical treatment. However, designing an efficient recommendation system is a complex task that requires significant effort. One of the significant challenges faced by such systems is the cold-start issue. Despite the increasing interest in recommender systems from both academia and industry and the extensive research to improve their performance, they still struggle to provide satisfactory recommendations for new users, i.e., cold-start users, with no historical interactions. The cross-domain recommendation is a promising approach to address the cold-start problem by transferring knowledge from an informative source domain to the target domain. Personalized bridge functions, which transfer knowledge from one domain to another, are more effective than using a common bridge in such situations. Our paper proposes a model, Deep Personalized Transfer of User Preferences for Cross-Domain Recommendations (DPTUPCDR), which utilizes Transfer Learning and Deep Neural Networks to enhance accuracy and address the cold-start challenge. Our model outperforms the state-of-the-art model based on two measures for evaluating prediction quality, demonstrating its effectiveness.

## Related Work
CMF: [Relational Learning via Collective Matrix Factorization Categories and Subject Descriptors (KDD 2008)](https://dl.acm.org/doi/pdf/10.1145/1401890.1401969?casa_token=S9kvmlp1bxEAAAAA:v96uHthvspO1ahgCZ1htH8sGl2voMvREqwXVYGf3X4WbvYXaD7tX1OsfXhx4k126HSOOtsbcbf9q)

EMCDR: [Cross-Domain Recommendation: An Embedding and Mapping Approach (IJCAI 2017)](https://www.ijcai.org/Proceedings/2017/0343.pdf)

PTUPCDR: [Personalized Transfer of User Preferences for Cross-domain Recommendation (WSDM 2022)](https://dl.acm.org/doi/pdf/10.1145/3488560.3498392?casa_token=fMj33BdRcdoAAAAA:7iA-ORhh02jV0wY2bPg3keZVcDxAXt5q8hM-9JM8oKrTFj7caBd-HUOICs6gfrIV6tch8NpcYYOC)

## Dataset

We conducted experiments on CDs and Vinyl, Movies and TV, and Books from the [Amazon Reviews Dataset](https://nijianmo.github.io/amazon/index.html) dataset, which consists of user reviews across multiple product domains.

## Requirements

To run this project, you will need the following environment and packages:

- Python 3.6 or higher
- PyTorch 1.0 or higher
- TensorFlow (any version compatible with the above Python and PyTorch versions)
- Pandas
- NumPy
- Tqdm

Please ensure that you have them installed before running the project. You can install them using `pip` with the following commands:

```bash
pip install torch>=1.0
pip install tensorflow
pip install pandas
pip install numpy
pip install tqdm
```

## Citation

We kindly ask you to cite our work if it aids or inspires your research:

```bibtex
@article{omidvar2023tackling,
  title={Tackling cold-start with deep personalized transfer of user preferences for cross-domain recommendation},
  author={Omidvar, Sepehr and Tran, Thomas},
  journal={International Journal of Data Science and Analytics},
  pages={1--10},
  year={2023},
  publisher={Springer}
}
```
