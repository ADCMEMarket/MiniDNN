#include <MiniDNN.h>
#include <fstream>

using namespace std;
using namespace MiniDNN;

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

int main()
{
    std::srand(123);
    
    Matrix x = Matrix::Zero(10, 100);
    ifstream ofile("../x.txt");
    for(int i=0;i<1000;i++){
        ofile >> x.data()[i];
    }
    ofile.close();

    double param[1102];
    ifstream ofile2("../nn.txt");
    for(int i=0;i<1102;i++){
        ofile2 >> param[i];
    }
    ofile2.close();

    Network net;
    Layer* layer1 = new FullyConnected<Tanh>(10, 20);
    Layer* layer2 = new FullyConnected<Tanh>(20, 20);
    Layer* layer3 = new FullyConnected<Tanh>(20, 20);
    Layer* layer4 = new FullyConnected<Identity>(20, 2);

    net.add_layer(layer1);
    net.add_layer(layer2);
    net.add_layer(layer3);
    net.add_layer(layer4);

    layer1->init();
    layer2->init();
    layer3->init();
    layer4->init();

    int n1 = layer1->set_parameters(param);
    int n2 = layer2->set_parameters(param+n1);
    int n3 = layer3->set_parameters(param+n1+n2);
    // std::cout << n1 << " " << n2 << " " << n3;
    int n4 = layer4->set_parameters(param+n1+n2+n3);


    Matrix pred = net.predict(x);
    printf("Prediction size = (%d, %d)\n", pred.rows(), pred.cols());
    ofstream ofile3("../out.txt");
    ofile3 << pred.transpose();
    ofile3.close();

    net.set_output(new TopGradients());
    Matrix top_grad(2, 100);
    for(int i=0;i<100;i++){
        top_grad(0,i) = 1.0;
        top_grad(1,i) = 0.0;
    }
    net.backprop<Matrix>(x, top_grad);
    auto V = net.get_sensitivity();
    ofstream ofile4("../g1_.txt");
    ofile4 << V;
    ofile4.close();

    for(int i=0;i<100;i++){
        top_grad(0,i) = 0.0;
        top_grad(1,i) = 1.0;
    }
    net.backprop<Matrix>(x, top_grad);
    V = net.get_sensitivity();
    ofstream ofile5("../g2_.txt");
    ofile5 << V;
    ofile5.close();

    // auto V = net.get_sensitivity();
    // std::cout << V.rows() << " " << V.cols() << std::endl;
    // std::cout << pred.cols() << std::endl;

    // auto derivs = net.get_derivatives();
    // for(auto &v1: derivs){
    //     std::cout << "*" << v1.size() << std::endl;
    // }

    return 0;
}