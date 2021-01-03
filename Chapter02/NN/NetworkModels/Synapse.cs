using System;

namespace NeuralNetwork.NetworkModels
{
    /// <summary>
    /// 突触(神经元中有许多小的结构体用于神经元间相互连接及信息传递,以及容纳权重和权重增量的容器，称之为神经突触)
    /// </summary>
    public class Synapse
	{
        #region -- Properties --


        /// <summary>
        /// Gets or sets the identifier.
        /// </summary>
        /// <value>
        /// The identifier.
        /// </value>
        public Guid Id { get; set; }


        /// <summary>
        /// 输入神经元
        /// </summary>
        /// <value>
        /// The input neuron.
        /// </value>
        public Neuron InputNeuron { get; set; }

        /// <summary>
        /// 输出神经元
        /// </summary>
        /// <value>
        /// The output neuron.
        /// </value>
        public Neuron OutputNeuron { get; set; }


        /// <summary>
        /// 权重
        /// </summary>
        /// <value>
        /// The weight.
        /// </value>
        public double Weight { get; set; }


        /// <summary>
        /// Delta权重 
        /// 神经网络最常见的学习规则之一就是所谓的delta规则。
        /// 这是一种监督规则，每次向网络呈现另一种学习模式时都会调用该规则，
        /// 每次发生这种情况都称为循环或轮次，每当输入模式通过一个或多个前向传播层，
        /// 然后通过一个或多个后向传播层，就会发生规则的调用。
        /// </summary>
        /// <value>
        /// The weight delta.
        /// </value>
        public double WeightDelta { get; set; }
        #endregion

        #region -- Constructor --


        /// <summary>
        /// Initializes a new instance of the <see cref="Synapse"/> class.
        /// </summary>
        public Synapse() { }


        /// <summary>
        /// Initializes a new instance of the <see cref="Synapse"/> class.
        /// </summary>
        /// <param name="inputNeuron">The input neuron.</param>
        /// <param name="outputNeuron">The output neuron.</param>
        public Synapse(Neuron inputNeuron, Neuron outputNeuron)
		{
			Id = Guid.NewGuid();
			InputNeuron = inputNeuron;
			OutputNeuron = outputNeuron;
			Weight = Network.GetRandom();
		}
		#endregion
	}
}