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
            mgr.SetupNetwork()
                .GetTrainingDataFromUser()
                .TrainNetworkToMinimum()
                .TestNetwork();

            Console.WriteLine("Press any key to train network for maximum");
            Console.ReadKey();

            mgr.SetupNetwork()
                .GetTrainingDataFromUser()
                .TrainNetworkToMaximum()
                .TestNetwork();

            Console.WriteLine("Press any key to exit");
            Console.ReadKey();


        }
    }
}
