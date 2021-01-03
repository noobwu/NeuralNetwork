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
		public List<HelperNeuron> InputLayer { get; set; }
		public List<List<HelperNeuron>> HiddenLayers { get; set; }
		public List<HelperNeuron> OutputLayer { get; set; }
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
