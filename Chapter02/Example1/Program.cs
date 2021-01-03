using Console = Colorful.Console;

namespace NeuralNetwork
{
    /// <summary>
    /// 
    /// </summary>
    public class Program
    {
        /// <summary>
        /// Defines the entry point of the application.
        /// </summary>
        /// <param name="args">The arguments.</param>
        static void Main(string[] args)
        {
            NNManager mgr = new NNManager();
            mgr.SetupNetwork() //创建网络
                .GetTrainingDataFromUser() //获取网络数据
                .TrainNetworkToMinimum()  //训练最小值
                .TestNetwork();  //测试

            Console.WriteLine("Press any key to train network for maximum");
            Console.ReadKey();

            mgr.SetupNetwork()  //创建网络
                .GetTrainingDataFromUser() //获取网络数据
                .TrainNetworkToMaximum() //训练最大值
                .TestNetwork(); //测试

            Console.WriteLine("Press any key to exit");
            Console.ReadKey();


        }
    }
}
