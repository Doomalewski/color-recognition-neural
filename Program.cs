using System;
using AForge.Neuro;
using AForge.Neuro.Learning;
using AForge.Math;
using AForge.Genetic;

internal class Program
{
    static double[] normalizeRGB(double[] rgb)
    {
        double[] normalized = new double[rgb.Length];
        for(int i=0;i<3;i++)
        {
            normalized[i] = rgb[i]/255;
        }
        return normalized;
    }
    static int IndexOfMaxElement(double[] output)
    {
        if (output == null || output.Length == 0)
        {
            throw new ArgumentException("Tablica nie może być pusta.");
        }

        double max = output[0];
        int index = 0;

        for (int i = 1; i < output.Length; i++)
        { 
            if(output[i] > max)
            {
                max = output[i];
                index = i;
            }
        }
        return index;
    }
    private static void Main(string[] args)
    {
        int patternSize = 3;
        int patterns = 24;



        int[][] input = new int[24][] {
         //shades of red
        new int[] {191, 49, 49},
        new int[] {155, 33, 33},
        new int[] {154, 22, 22},
        new int[] {235, 43, 139},
        new int[] {208, 96, 96},
        new int[] {183, 47, 92},
        new int[] {141, 24, 24},
        new int[] {247, 23 ,23},
        //shades of green
        new int[] {25, 201, 89},
        new int[] {65, 201, 114},
        new int[] {35, 129, 70},
        new int[] {84, 196, 91},
        new int[] {4, 64, 14 },
        new int[] {140, 240, 180},
        new int[] {60, 125, 86},
        new int[] {46, 192, 105 },
        //shades of blue
        new int[] {43, 49, 235},
        new int[] {30, 34, 168},
        new int[] {91, 96, 222},
        new int[] {53, 21, 213},
        new int[] {21, 162, 213},
        new int[] {72, 190, 233},
        new int[] {28, 121, 155},
        new int[] {25, 130, 201},};
        double[][] output = new double[24][] {
            new double[] { 1, 0, 0 },
            new double[] { 1, 0, 0 },
            new double[] { 1, 0, 0 },
            new double[] { 1, 0, 0 },
            new double[] { 1, 0, 0 },
            new double[] { 1, 0, 0 },
            new double[] { 1, 0, 0 },
            new double[] { 1, 0, 0 },
            new double[] { 0, 1, 0 },
            new double[] { 0, 1, 0 },
            new double[] { 0, 1, 0 },
            new double[] { 0, 1, 0 },
            new double[] { 0, 1, 0 },
            new double[] { 0, 1, 0 },
            new double[] { 0, 1, 0 },
            new double[] { 0, 1, 0 },
            new double[] { 0, 0, 1 },
            new double[] { 0, 0, 1 },
            new double[] { 0, 0, 1 },
            new double[] { 0, 0, 1 },
            new double[] { 0, 0, 1 },
            new double[] { 0, 0, 1 },
            new double[] { 0, 0, 1 },
            new double[] { 0, 0, 1 },};

        double[][] normalizedInput = new double[patterns][];
        for (int i = 0; i < patterns; i++)
        {
            normalizedInput[i] = new double[patternSize];
            for (int j = 0; j < patternSize; j++)
            {
                normalizedInput[i][j] = input[i][j] / 255.0;
            }
        }

        BipolarSigmoidFunction activationFunction = new BipolarSigmoidFunction(2.0f);

        ActivationNetwork neuralNet = new ActivationNetwork(activationFunction, patternSize, 10, 3);

        BackPropagationLearning teacher = new BackPropagationLearning(neuralNet)
        {
            LearningRate = 0.1,
            Momentum = 0.9
        };
        int epoch = 0;
        double error = double.MaxValue;
        Console.WriteLine("=======================================Teaching...=======================================");
        while (error > 1)
        {
            epoch++;
            error = teacher.RunEpoch(normalizedInput, output);
            if (epoch % 100 == 0)
            {
                Console.WriteLine($"Epoch: {epoch}, Error: {error}");
            }
        }


        int r, g, b;

        Console.Write("Provide red value: ");
        r = int.Parse(Console.ReadLine());

        Console.Write("Provide green value: ");
        g = int.Parse(Console.ReadLine());

        Console.Write("Provide blue value: ");
        b = int.Parse(Console.ReadLine());

        Console.WriteLine($"You provided: r={r}, g={g}, b={b}");

        double[] inputTest = new double[] { r, g, b };
        double[] normalizedInputTest = normalizeRGB(inputTest);
        double[] outputTest = neuralNet.Compute(normalizedInputTest);
        Console.WriteLine("=======================================Calculated probability=======================================");
        Console.WriteLine($"Normalised Input: {string.Join(", ", normalizedInputTest)} -> Output: {string.Join(", ", outputTest)}");
        Console.WriteLine("=======================================Calculated probability=======================================");

        switch (IndexOfMaxElement(outputTest)){
            case 0:
                Console.WriteLine("Your colour is Red");
                break;

            case 1:
                Console.WriteLine("Your color is Green");
                break;

            case 2: Console.WriteLine("Your color is Blue");
                break;
        }

    }
}