# python-neural-networks-water
A project to train neural network to check if water is potable to drink

Based on 9 input variables:
ph, hardness, solids , chloramines, sulfate, conductivity, organic_carbon, trihalomethanes and turbidity
create neural network which will be clasify water to two groups: 0 - unpotible and 1 - potable to drink

Aim of project is to analyze impact hyperparameters for level of accuracy like:
- number of layers
- number of neurons
- type of activation function

After checking for 2, 3 and 4  hidden layers, the best accuracy occured for 2 layers. For 3 and 4 layers was overfitting.

Image for 2 layers.

![obraz](https://user-images.githubusercontent.com/44522588/226682650-55978aba-dcd7-4ed1-8cc1-c64afd06c0a3.png)

After checking various combination of neuron numbers the best result were for number of neurons: 30, 30.

![obraz](https://user-images.githubusercontent.com/44522588/226684450-549d965d-63bf-4f37-8e92-5195f9fc5179.png)

After checking types of activation function like: ReLU, Sigmoid and Tanh the best result was for ReLU.

CONCLUSIONS:
Accuracy was around 70%. Data are from Kaggle, To compare other people get similar results around 70%, the best result was 80%. Data are random so results does not make sense. 
There were not enough data to train better this model.
