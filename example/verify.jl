using ADCME
using DelimitedFiles

reset_default_graph()
# n = ae_num([10,20,20,20,2])
θ = constant(ae_init([10,20,20,20,2], method="xavier"))
x = constant(rand(100,10))
# y = ae(x, [20,20,20,2], θ)

y = ae(x, [20,20,20,2], θ)

g1 = tf.gradients(y[:,1], x)[1]
g2 = tf.gradients(y[:,2], x)[1]

sess = Session(); init(sess)
run(sess, y)
writedlm("nn.txt", run(sess, θ))
writedlm("x.txt", run(sess, x)'[:])
writedlm("y.txt", run(sess, y))
writedlm("g1.txt", run(sess, g1'))
writedlm("g2.txt", run(sess, g2'))

