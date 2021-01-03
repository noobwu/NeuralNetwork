using System;
using System.Collections.Generic;

namespace NeuralNetwork.Helpers
{
    /// <summary>
    /// 
    /// </summary>
    public class HelperNetwork
	{
		/// <summary>
		///学习率(学习率通过控制权重大小和学习过程中网络的偏差变化来改变系统的整体学习速度)
		/// </summary>
		/// <value>
		/// The learning rate.
		/// </value>
		public double LearningRate { get; set; }
		/// <summary>
		/// 动量(动量仅将先前权重更新的一小部分添加到当前权重更新中，动量用于防止系统收敛于局部最小值而非全局最小值)
		/// </summary>
		/// <value>
		/// The momentum.
		/// </value>
		public double Momentum { get; set; }
		/// <summary>
		/// 输入层（这是网络的初始数据。对于每个输入，其输出到隐藏层的值是初始输入值）
		/// </summary>
		/// <value>
		/// The input layer.
		/// </value>
		public List<HelperNeuron> InputLayer { get; set; }
		/// <summary>
		///  隐藏层（这是网络的核心和灵魂，也是程序发挥魔力的根本。该层中的神经元为每个输入配置权重。
		///  这些权重随机设置初始值，并在网络训练时进行调整，以使神经元的输出更接近预期结果（如果幸运的话） ）
		/// </summary>
		/// <value>
		/// The hidden layers.
		/// </value>
		public List<List<HelperNeuron>> HiddenLayers { get; set; }
		/// <summary>
		/// 输出层(这是神经网络在执行计算后得到的输出。简单案例中的输出将设置为true， false，或者on，off。
		/// 神经元为每个输入配置权重，这些输入来自先前的隐藏层。
		/// 虽然通常只有一个输出神经元，但如果需要或想要多个输出神经元，也可以设置更多神经元。)
		/// </summary>
		/// <value>
		/// The output layer.
		/// </value>
		public List<HelperNeuron> OutputLayer { get; set; }
		/// <summary>
		/// 突触(神经元中有许多小的结构体用于神经元间相互连接及信息传递,以及容纳权重和权重增量的容器，称之为神经突触)
		/// </summary>
		/// <value>
		/// The synapses.
		/// </value>
		public List<HelperSynapse> Synapses { get; set; }

		public HelperNetwork()
		{
			InputLayer = new List<HelperNeuron>();
			HiddenLayers = new List<List<HelperNeuron>>();
			OutputLayer = new List<HelperNeuron>();
			Synapses = new List<HelperSynapse>();
		}
	}

	public class HelperNeuron
	{
		public Guid Id { get; set; }
		public double Bias { get; set; }
		public double BiasDelta { get; set; }
		public double Gradient { get; set; }
		public double Value { get; set; }
	}

	public class HelperSynapse
	{
		public Guid Id { get; set; }
		public Guid OutputNeuronId { get; set; }
		public Guid InputNeuronId { get; set; }
		public double Weight { get; set; }
		public double WeightDelta { get; set; }
	}
}
