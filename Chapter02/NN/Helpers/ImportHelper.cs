using System;
using System.Collections.Generic;
using System.Windows.Forms;
using System.IO;
using System.Linq;
using NeuralNetwork.NetworkModels;
using Newtonsoft.Json;

namespace NeuralNetwork.Helpers
{
    /// <summary>
    ///  导入相关操作
    /// </summary>
    public static class ImportHelper
	{
        /// <summary>
        /// 导入现有网络
        /// </summary>
        /// <returns></returns>
        public static Network ImportNetwork()
		{
			var dn = GetHelperNetwork();
			if (dn == null) 
			    return null;

			var network = new Network();//新网络
			var allNeurons = new List<Neuron>();//神经元列表

			network.LearningRate = dn.LearningRate;//学习率
			network.Momentum = dn.Momentum;//动量

			//导入输入层
			foreach (var n in dn.InputLayer)
			{
				var neuron = new Neuron
				{
					Id = n.Id,
					Bias = n.Bias,
					BiasDelta = n.BiasDelta,
					Gradient = n.Gradient,
					Value = n.Value
				};

				network.InputLayer?.Add(neuron);
				allNeurons.Add(neuron);
			}

			//导入隐藏层
			foreach (var layer in dn.HiddenLayers)
			{
				var neurons = new List<Neuron>();
				foreach (var n in layer)
				{
					var neuron = new Neuron
					{
						Id = n.Id,
						Bias = n.Bias,
						BiasDelta = n.BiasDelta,
						Gradient = n.Gradient,
						Value = n.Value
					};

					neurons.Add(neuron);
					allNeurons.Add(neuron);
				}

				network.HiddenLayers?.Add(neurons);
			}

		    //导入输出层
			foreach (var n in dn.OutputLayer)
			{
				var neuron = new Neuron
				{
					Id = n.Id,
					Bias = n.Bias,
					BiasDelta = n.BiasDelta,
					Gradient = n.Gradient,
					Value = n.Value
				};

				network.OutputLayer?.Add(neuron);
				allNeurons.Add(neuron);
			}

			//Synapses
			foreach (var syn in dn.Synapses)
			{
				var synapse = new Synapse { Id = syn.Id };
				var inputNeuron = allNeurons.First(x => x.Id == syn.InputNeuronId);
				var outputNeuron = allNeurons.First(x => x.Id == syn.OutputNeuronId);
				synapse.InputNeuron = inputNeuron;
				synapse.OutputNeuron = outputNeuron;
				synapse.Weight = syn.Weight;
				synapse.WeightDelta = syn.WeightDelta;

				inputNeuron?.OutputSynapses?.Add(synapse);
				outputNeuron?.InputSynapses?.Add(synapse);
			}

			return network;
		}

		public static List<NNDataSet> ImportDatasets(string name)
		{
			try
			{
				using (var file = File.OpenText(name))
				{
					return JsonConvert.DeserializeObject<List<NNDataSet>>(file.ReadToEnd());
				}
			}
			catch (Exception)
			{
				return null;
			}
		}

        /// <summary>
        /// Gets the helper network.
        /// </summary>
        /// <returns></returns>
        private static HelperNetwork GetHelperNetwork()
		{
			try
			{
				var dialog = new OpenFileDialog
				{
					Multiselect = false,
					Title = "Open Network File",
					Filter = "Text File|*.txt;"
				};

				using (dialog)
				{
					if (dialog.ShowDialog() != DialogResult.OK) return null;

					using (var file = File.OpenText(dialog.FileName))
					{
						return JsonConvert.DeserializeObject<HelperNetwork>(file.ReadToEnd());
					}
				}
			}
			catch (Exception)
			{
				return null;
			}
		}
	}
}
