using System;

namespace NeuralNetwork.NetworkModels
{
    /// <summary>
    /// Sigmoid函数
    /// </summary>
    public static class Sigmoid
	{

        /// <summary>
        /// 神经元的输出=Y=f(w1*x1+w2*x2+b)
        /// </summary>
        /// <param name="x">The x.</param>
        /// <returns></returns>
        public static double Output(double x)
		{
			return x < -45.0 ? 0.0 : x > 45.0 ? 1.0 : 1.0 / (1.0 + Math.Exp(-x));
		}

        /// <summary>
        /// 导数
        /// </summary>
        /// <param name="x">The x.</param>
        /// <returns></returns>
        public static double Derivative(double x)
		{
			return x * (1 - x);
		}
	}
}