# Task-Adaptive-Meta-Learning

AAAI-23 Paper: [Task-Adaptive Meta-Learning Framework for Advancing Spatial Generalizability](https://ojs.aaai.org/index.php/AAAI/article/view/26680)

Abstract: Spatio-temporal machine learning is critically needed for a variety of societal applications, such as agricultural monitoring, hydrological forecast, and traffic management. These applications greatly rely on regional features that characterize spatial and temporal differences. However, spatio-temporal data often exhibit complex patterns and significant data variability across different locations. The labels in many real-world applications can also be limited, which makes it difficult to separately train independent models for different locations. Although meta learning has shown promise in model adaptation with small samples, existing meta learning methods remain limited in handling a large number of heterogeneous tasks, e.g., a large number of locations with varying data patterns. To bridge the gap, we propose task-adaptive formulations and a model-agnostic meta-learning framework that ensembles regionally heterogeneous data into location-sensitive meta tasks. We conduct task adaptation following an easy-to-hard task hierarchy in which different meta models are adapted to tasks of different difficulty levels. One major advantage of our proposed method is that it improves the model adaptation to a large number of heterogeneous tasks. It also enhances the model generalization by automatically adapting the meta model of the corresponding difficulty level to any new tasks. We demonstrate the superiority of our proposed framework over a diverse set of baselines and state-of-the-art meta-learning frameworks. Our extensive experiments on real crop yield data show the effectiveness of the proposed method in handling spatial-related heterogeneous tasks in real societal applications.

# Citation
```angular2html
@article{Liu_Liu_Xie_Jin_Jia_2023, 
        title={Task-Adaptive Meta-Learning Framework for Advancing Spatial Generalizability}, 
        volume={37}, 
        url={https://ojs.aaai.org/index.php/AAAI/article/view/26680}, 
        DOI={10.1609/aaai.v37i12.26680}, 
        number={12}, 
        journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
        author={Liu, Zhexiong and Liu, Licheng and Xie, Yiqun and Jin, Zhenong and Jia, Xiaowei}, 
        year={2023}, 
        month={Jun.}, 
        pages={14365-14373}}
```