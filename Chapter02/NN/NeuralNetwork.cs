using System;
using System.Collections.Generic;
using System.Linq;
using Console = Colorful.Console;

namespace NeuralNetwork
{
    using System.Drawing;
    using Helpers;
    using NetworkModels;

    /// <summary>
    /// 神经网络相关操作
    /// </summary>
    public class NNManager
    {
        #region -- Variables --       
        /// <summary>
        /// 输入参数
        /// </summary>
        private int _numInputParameters;
        /// <summary>
        /// 隐藏层
        /// </summary>
        private int _numHiddenLayers;
        /// <summary>
        /// 隐藏神经元
        /// </summary>
        private int[] _hiddenNeurons;
        /// <summary>
        /// 输出参数
        /// </summary>
        private int _numOutputParameters;
        /// <summary>
        /// 网络
        /// </summary>
        private Network _network;
        /// <summary>
        ///  数据集
        /// </summary>
        private List<NNDataSet> _dataSets;
        #endregion

        /// <summary>
        /// 创建新网络
        /// </summary>
        /// <returns></returns>
        public NNManager SetupNetwork()
        {
            _numInputParameters = 2;

            int[] hidden = new int[2];
            hidden[0] = 3;
            hidden[1] = 1;
            _numHiddenLayers = 1;
            _hiddenNeurons = hidden;
            _numOutputParameters = 1;
            _network = new Network(_numInputParameters, _hiddenNeurons, _numOutputParameters);
            return this;
        }


        /// <summary>
        /// Sets the number input parameters.
        /// </summary>
        /// <param name="num">The number.</param>
        /// <returns></returns>
        public NNManager SetNumInputParameters(int num)
        {
            _numInputParameters = num;
            return this;
        }


        /// <summary>
        /// Sets the number neurons in hidden layer.
        /// </summary>
        /// <param name="hiddenNeurons">The hidden neurons.</param>
        /// <param name="numHiddenLayers">The number hidden layers.</param>
        /// <returns></returns>
        public NNManager SetNumNeuronsInHiddenLayer(int[] hiddenNeurons, int numHiddenLayers = 1)
        {
            _numHiddenLayers = numHiddenLayers;
            _hiddenNeurons = hiddenNeurons;
            return this;
        }


        /// <summary>
        /// Sets the number output parameters.
        /// </summary>
        /// <param name="numOutputs">The number outputs.</param>
        /// <returns></returns>
        public NNManager SetNumOutputParameters(int numOutputs = 1)
        {
            _numOutputParameters = numOutputs;
            return this;
        }


        /// <summary>
        /// Gets the training data from user.
        /// </summary>
        /// <returns></returns>
        public NNManager GetTrainingDataFromUser()
        {
            PrintNewLine();

            var numDataSets = GetInput("\tHow many datasets are you going to enter? ", 1, int.MaxValue);

            var newDatasets = new List<NNDataSet>();
            for (var i = 0; i < numDataSets; i++)
            {
                var values = GetInputData($"\tData Set {i + 1}: ");
                if (values == null)
                {
                    PrintNewLine();
                    return this;
                }

                var expectedResult = GetExpectedResult($"\tExpected Result for Data Set {i + 1}: ");
                if (expectedResult == null)
                {
                    PrintNewLine();
                    return this;
                }

                newDatasets.Add(new NNDataSet(values, expectedResult));
            }

            _dataSets = newDatasets;
            PrintNewLine();
            return this;
        }

        /// <summary>
        /// Gets the input data.
        /// </summary>
        /// <param name="message">The message.</param>
        /// <returns></returns>
        public double[] GetInputData(string message)
        {
            Console.Write(message, Color.Yellow);
            var line = GetLine();

            if (line.Equals("menu", StringComparison.InvariantCultureIgnoreCase))
                return null;

            while (line == null || line.Split(' ').Length != _numInputParameters)
            {
                Console.WriteLine($"\t{_numInputParameters} inputs are required.", Color.Red);
                PrintNewLine();
                Console.WriteLine(message);
                line = GetLine();
            }

            var values = new double[_numInputParameters];
            var lineNums = line.Split(' ');
            for (var i = 0; i < lineNums.Length; i++)
            {
                if (double.TryParse(lineNums[i], out var num))
                {
                    values[i] = num;
                }
                else
                {
                    Console.WriteLine("\tYou entered an invalid number.  Try again", Color.Red);
                    PrintNewLine(2);
                    return GetInputData(message);
                }
            }

            return values;
        }


        /// <summary>
        /// Gets the expected result.
        /// </summary>
        /// <param name="message">The message.</param>
        /// <returns></returns>
        public double[] GetExpectedResult(string message)
        {
            Console.Write(message, Color.Yellow);
            var line = GetLine();

            if (line != null && line.Equals("menu", StringComparison.InvariantCultureIgnoreCase))
                return null;

            while (line == null || line.Split(' ').Length != _numOutputParameters)
            {
                Console.WriteLine($"\t{_numOutputParameters} outputs are required.", Color.Red);
                PrintNewLine();
                Console.WriteLine(message);
                line = GetLine();
            }

            var values = new double[_numOutputParameters];
            var lineNums = line.Split(' ');
            for (var i = 0; i < lineNums.Length; i++)
            {
                if (int.TryParse(lineNums[i], out var num) && (num == 0 || num == 1))
                {
                    values[i] = num;
                }
                else
                {
                    Console.WriteLine("\tYou must enter 1s and 0s!", Color.Red);
                    PrintNewLine(2);
                    return GetExpectedResult(message);
                }
            }

            return values;
        }


        /// <summary>
        /// Tests the network.
        /// </summary>
        /// <returns></returns>
        public NNManager TestNetwork()
        {
            Console.WriteLine("\tTesting Network", Color.Yellow);
            PrintNewLine();

            while (true)
            {
                PrintUnderline(50);
                var values = GetInputData($"\tType {_numInputParameters} inputs: ");
                if (values == null)
                {
                    PrintNewLine();
                    return this;
                }

                var results = _network?.Compute(values);
                PrintNewLine();

                foreach (var result in results)
                {
                    Console.WriteLine("\tOutput: " + DoubleConverter.ToExactString(result), Color.Aqua);
                }

                PrintNewLine();
                return this;
            }
        }

        /// <summary>
        /// Trains the network to minimum.
        /// </summary>
        /// <returns></returns>
        public NNManager TrainNetworkToMinimum()
        {
            var minError = GetDouble("\tMinimum Error: ", 0.000000001, 1.0);
            PrintNewLine();
            Console.WriteLine("\tTraining...");
            _network.Train(_dataSets, minError);
            Console.WriteLine("\t**Training Complete**", Color.Green);
            PrintNewLine();
            return this;
        }


        /// <summary>
        /// Trains the network to maximum.
        /// </summary>
        /// <returns></returns>
        public NNManager TrainNetworkToMaximum()
        {
            var maxEpoch = GetInput("\tMax Epoch: ", 1, int.MaxValue);
            if (!maxEpoch.HasValue)
            {
                PrintNewLine();
                return this;
            }
            PrintNewLine();
            Console.WriteLine("\tTraining...");
            _network.Train(_dataSets, maxEpoch.Value);
            Console.WriteLine("\t**Training Complete**", Color.Green);
            PrintNewLine();
            return this;
        }
        /// <summary>
        /// Trains the network.
        /// </summary>
        /// <returns></returns>
        public NNManager TrainNetwork()
        {
            Console.WriteLine("Network Training", Color.Yellow);
            PrintUnderline(50);
            Console.WriteLine("\t1. Train to minimum error", Color.Yellow);
            Console.WriteLine("\t2. Train to max epoch", Color.Yellow);
            Console.WriteLine("\t3. Network Menu", Color.Yellow);
            PrintNewLine();

            switch (GetInput("\tYour Choice: ", 1, 3))
            {
                case 1:
                    TrainNetworkToMinimum();
                    break;
                case 2:
                    TrainNetworkToMaximum();
                    break;
                case 3:
                    break;
            }
            PrintNewLine();
            return this;
        }

        /// <summary>
        /// 导入现在有网络
        /// </summary>
        /// <returns></returns>
        public NNManager ImportNetwork()
        {
            PrintNewLine();
            _network = ImportHelper.ImportNetwork();
            if (_network == null)
            {
                WriteError("\t****Something went wrong while importing your network.****");
                return this;
            }

            _numInputParameters = _network.InputLayer.Count;
            _hiddenNeurons = new int[_network.HiddenLayers.Count];
            _numOutputParameters = _network.OutputLayer.Count;

            Console.WriteLine("\t**Network successfully imported.**", Color.Green);
            PrintNewLine();
            return this;
        }

        /// <summary>
        /// Exports the network.
        /// </summary>
        /// <returns></returns>
        public NNManager ExportNetwork()
        {
            PrintNewLine();
            Console.WriteLine("\tExporting Network...");
            ExportHelper.ExportNetwork(_network);
            Console.WriteLine("\t**Exporting Complete!**", Color.Green);
            PrintNewLine();
            return this;
        }

        /// <summary>
        /// Imports the datasets.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <returns></returns>
        public NNManager ImportDatasets(string name)
        {
            PrintNewLine();
            _dataSets = ImportHelper.ImportDatasets(name);
            if (_dataSets == null)
            {
                WriteError("\t--Something went wrong while importing your datasets.--");
                return this;
            }

            if (_dataSets.Any(x => x.Values.Length != _numInputParameters || _dataSets.Any(y => y.Targets.Length != _numOutputParameters)))
            {
                WriteError($"\t--The dataset does not fit the network.  Network requires datasets that have {_numInputParameters} inputs and {_numOutputParameters} outputs.--");
                return this;
            }

            Console.WriteLine("\t**Datasets successfully imported.**", Color.Green);
            PrintNewLine();
            return this;
        }


        /// <summary>
        /// Exports the datasets.
        /// </summary>
        /// <returns></returns>
        public NNManager ExportDatasets()
        {
            PrintNewLine();
            Console.WriteLine("\tExporting Datasets...");
            ExportHelper.ExportDatasets(_dataSets);
            Console.WriteLine("\t**Exporting Complete!**", Color.Green);
            PrintNewLine();
            return this;
        }


        /// <summary>
        /// Gets the line.
        /// </summary>
        /// <returns></returns>
        public string GetLine()
        {
            var line = Console.ReadLine();
            return line?.Trim() ?? string.Empty;
        }


        /// <summary>
        /// Gets the input.
        /// </summary>
        /// <param name="message">The message.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns></returns>
        public int? GetInput(string message, int min, int max)
        {
            Console.Write(message);
            var num = GetNumber();
            if (!num.HasValue) return null;

            while (!num.HasValue || num < min || num > max)
            {
                Console.Write(message, Color.Red);
                num = GetNumber();
            }

            return num.Value;
        }

        /// <summary>
        /// Gets the double.
        /// </summary>
        /// <param name="message">The message.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns></returns>
        public double GetDouble(string message, double min, double max)
        {
            Console.Write(message);
            var num = GetDouble();

            while (num < min || num > max)
            {
                Console.Write(message, Color.Red);
                num = GetDouble();

            }

            return num;
        }


        /// <summary>
        /// Gets the array input.
        /// </summary>
        /// <param name="message">The message.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="numToGet">The number to get.</param>
        /// <returns></returns>
        public int[] GetArrayInput(string message, int min, int numToGet)
        {
            var nums = new int[numToGet];

            for (var i = 0; i < numToGet; i++)
            {
                Console.Write(message + " " + (i + 1) + ": ");
                var num = GetNumber();

                while (!num.HasValue || num < min)
                {
                    Console.Write(message + " " + (i + 1) + ": ");
                    num = GetNumber();
                }

                nums[i] = num.Value;
            }

            return nums;
        }


        /// <summary>
        /// Gets the number.
        /// </summary>
        /// <returns></returns>
        public int? GetNumber()
        {
            var line = GetLine();

            if (line.Equals("menu", StringComparison.InvariantCultureIgnoreCase))
                return null;

            return int.TryParse(line, out var num) ? num : 0;
        }


        /// <summary>
        /// Gets the double.
        /// </summary>
        /// <returns></returns>
        public double GetDouble()
        {
            var line = GetLine();
            return line != null && double.TryParse(line, out var num) ? num : 0;
        }


        /// <summary>
        /// Prints the new line.
        /// </summary>
        /// <param name="numNewLines">The number new lines.</param>
        public void PrintNewLine(int numNewLines = 1)
        {
            for (var i = 0; i < numNewLines; i++)
                Console.WriteLine();
        }


        /// <summary>
        /// Prints the underline.
        /// </summary>
        /// <param name="numUnderlines">The number underlines.</param>
        public void PrintUnderline(int numUnderlines)
        {
            for (var i = 0; i < numUnderlines; i++)
                Console.Write("-", Color.Yellow);
            PrintNewLine(2);
        }

        /// <summary>
        /// Writes the error.
        /// </summary>
        /// <param name="error">The error.</param>
        public void WriteError(string error)
        {
            Console.WriteLine(error, Color.Red);
            Exit();
        }

        /// <summary>
        /// Exits this instance.
        /// </summary>
        public void Exit()
        {
            Console.WriteLine("Exiting...", Color.Yellow);
            Console.ReadLine();
            Environment.Exit(0);
        }

    }
}
