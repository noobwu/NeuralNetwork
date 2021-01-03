using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.NetworkModels
{
    using EnsureThat;

    /// <summary>
    /// 神经元(神经元是神经网络中计算的基本单位)
    /// 神经元有时也被称为节点或单元。它接收来自其他节点或外部源的输入并计算输出。
    /// 每个输入都有一个相关的权重（见图1—1中的w1和w2） ，它是根据输入信息的相对重要性进行分配的。
    /// 节点将输入的加权和提供给函数 （激活函数，后面将详细介绍）。
    /// 虽然这是对神经元是什么以及它能做什么的极端简化，但基本上就是这样。
    /// </summary>
    public class Neuron
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
        /// 输入突触
        /// </summary>
        /// <value>
        /// The input synapses.
        /// </value>
        public List<Synapse> InputSynapses { get; set; }


        /// <summary>
        /// 输出突触
        /// </summary>
        /// <value>
        /// The output synapses.
        /// </value>
        public List<Synapse> OutputSynapses { get; set; }


        /// <summary>
        /// 偏差
        /// </summary>
        /// <value>
        /// The bias.
        /// </value>
        public double Bias { get; set; }


        /// <summary>
        /// 偏差Delta
        /// </summary>
        /// <value>
        /// The bias delta.
        /// </value>
        public double BiasDelta { get; set; }

        /// <summary>
        ///  梯度
        /// </summary>
        /// <value>
        /// The gradient.
        /// </value>
        public double Gradient { get; set; }


        /// <summary>
        /// Gets or sets the value.
        /// </summary>
        /// <value>
        /// The value.
        /// </value>
        public double Value { get; set; }


        /// <summary>
        /// Gets or sets a value indicating whether this instance is mirror.
        /// </summary>
        /// <value>
        ///   <c>true</c> if this instance is mirror; otherwise, <c>false</c>.
        /// </value>
        public bool IsMirror { get; set; }


        /// <summary>
        /// Gets or sets a value indicating whether this instance is canonical.
        /// </summary>
        /// <value>
        ///   <c>true</c> if this instance is canonical; otherwise, <c>false</c>.
        /// </value>
        public bool IsCanonical { get; set; }
        #endregion

        #region -- Constructors --


        /// <summary>
        /// Initializes a new instance of the <see cref="Neuron"/> class.
        /// </summary>
        public Neuron()
		{
			Id = Guid.NewGuid();
			InputSynapses = new List<Synapse>();
			OutputSynapses = new List<Synapse>();
			Bias = Network.GetRandom();
		}


        /// <summary>
        /// Initializes a new instance of the <see cref="Neuron"/> class.
        /// </summary>
        /// <param name="inputNeurons">The input neurons.</param>
        public Neuron(IEnumerable<Neuron> inputNeurons) : this()
		{
		    Ensure.That(inputNeurons).IsNotNull();

			foreach (var inputNeuron in inputNeurons)
			{
				var synapse = new Synapse(inputNeuron, this);
				inputNeuron?.OutputSynapses?.Add(synapse);
				InputSynapses?.Add(synapse);
			}
		}
        #endregion

        #region -- Values & Weights --


        /// <summary>
        /// 在神经网络中，向前传播数据以获取输出，然后将其与实际预期值进行比较以获得误差，
        /// 这是正确数据与机器学习算法预测数据之间的差异。为了最小化该误差，
        /// 现在你必须求每个权重的误差导数来向后传播，然后从权重中减去该误差导数值。
        /// </summary>
        /// <returns></returns>
        public virtual double CalculateValue()
		{
			return Value = Sigmoid.Output(InputSynapses.Sum(a => a.Weight * a.InputNeuron.Value) + Bias);
		}

        /// <summary>
        /// 计算误差(正确数据与机器学习算法预测数据之间的差异)
        /// </summary>
        /// <param name="target">The target.</param>
        /// <returns></returns>
        public double CalculateError(double target)
		{
			return target - Value;
		}


        /// <summary>
        /// Calculates the gradient.
        /// </summary>
        /// <param name="target">The target.</param>
        /// <returns></returns>
        public double CalculateGradient(double? target = null)
		{
            if (target == null)
            {
                return Gradient = OutputSynapses.Sum(a => a.OutputNeuron.Gradient * a.Weight) * Sigmoid.Derivative(Value);
            }
			return Gradient = CalculateError(target.Value) * Sigmoid.Derivative(Value);
		}


        /// <summary>
        /// Updates the weights.
        /// </summary>
        /// <param name="learnRate">The learn rate.</param>
        /// <param name="momentum">The momentum.</param>
        public void UpdateWeights(double learnRate, double momentum)
		{
			var prevDelta = BiasDelta;
			BiasDelta = learnRate * Gradient;
			Bias += BiasDelta + momentum * prevDelta;

			foreach (var synapse in InputSynapses)
			{
				prevDelta = synapse.WeightDelta;
				synapse.WeightDelta = learnRate * Gradient * synapse.InputNeuron.Value;
				synapse.Weight += synapse.WeightDelta + momentum * prevDelta;
			}
		}
		#endregion
	}
}