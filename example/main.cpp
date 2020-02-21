#include <MiniDNN.h>

using namespace MiniDNN;

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

int main()
{
    std::srand(123);
    Matrix x = Matrix::Random(400, 100);
    Matrix top_grad = Matrix::Random(2,100);
    Network net;

    Layer* layer1 = new FullyConnected<Tanh>(400, 20);
    Layer* layer2 = new FullyConnected<Tanh>(20, 20);
    Layer* layer3 = new FullyConnected<Tanh>(20, 20);
    Layer* layer4 = new FullyConnected<Tanh>(20, 20);
    Layer* layer5 = new FullyConnected<Identity>(20, 2);

    // Add layers to the network object
    net.add_layer(layer1);
    net.add_layer(layer2);
    net.add_layer(layer3);
    net.add_layer(layer4);
    net.add_layer(layer5);

    net.init(0, 0.01, 123);

    Matrix pred = net.predict(x);
    net.set_output(new TopGradients());
    net.backprop<Matrix>(x, top_grad);

    auto V = net.get_sensitivity();
    std::cout << V.rows() << " " << V.cols() << std::endl;
    std::cout << pred.cols() << std::endl;

    auto derivs = net.get_derivatives();
    for(auto &v1: derivs){
        std::cout << v1.size() << std::endl;
    }

    return 0;
}