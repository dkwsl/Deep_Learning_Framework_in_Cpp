#include <cstdio>
#include <cstdlib>
#include <memory>
#include "module/module.h"
#include "module/loss.h"
#include "module/optimizer.h"
#include "module/mlp.h"

const unsigned int seed = 114514;
const double eps = 1e-4;

namespace XorDataset
{
	fish::MLP net;
	//fish::SGD optim;
	//fish::AdaGrad optim;
	fish::RMSProp optim;

	const int epoch = 1000000;
	const double learn_rate = 0.001;
	const double momentum = 0.9;
	const double weight_decay = 0.0001;

	struct Data
	{
		std::shared_ptr<fish::Tensor> input, output;
		
		Data() {}
		Data(std::shared_ptr<fish::Tensor> _input, std::shared_ptr<fish::Tensor> _output) : input(_input), output(_output) {}

	};
	std::vector<Data> dataset, testset;
	void buildDataset()
	{
		const int data_len = 1;
		for(int x = 0; x < (1 << data_len); x++)
		{
			for(int y = 0; y < (1 << data_len); y++)
			{
				dataset.emplace_back(Data(std::make_shared<fish::Tensor>(fish::Tensor({x, y}, {2, 1})),
				                          std::make_shared<fish::Tensor>(fish::Tensor({x ^ y}, {1, 1}))));
			}
		}
		for(int i = 1; i <= 20; i++)
		{
			int x = rand() % (1 << data_len), y = rand() % (1 << data_len);
			testset.emplace_back(Data(std::make_shared<fish::Tensor>(fish::Tensor({x, y}, {2, 1})),
			                          std::make_shared<fish::Tensor>(fish::Tensor({x ^ y}, {1, 1}))));
		}
	}

	void init()
	{
		net = fish::MLP({2, 64, 64, 1});
		//optim = fish::SGD(net.collectParameters(), learn_rate, momentum, weight_decay);
		//optim = fish::AdaGrad(net.collectParameters(), learn_rate);
		optim = fish::RMSProp(net.collectParameters(), learn_rate);
	}

	void run()
	{
		buildDataset();
		init();


		// 训练
		for(int i = 1; i <= epoch; i++)
		{
			double epoch_loss = 0.0;
			for(int j = 0; j < dataset.size(); j++)
			{
				optim.clearGrad();

				auto x = net.forward(dataset[j].input);

				auto loss = fish::MSELoss(x, dataset[j].output);
				epoch_loss += loss->data[0];

				fish::backward(loss);

				optim.update();
			}
			epoch_loss /= dataset.size();
			printf("epoch %d/%d, loss: %f\n", i, epoch, epoch_loss);

			if(epoch_loss < eps) break;
		}

		getchar();

		// 测试
		fish::disableAutograd();
		for(int i = 0; i < testset.size(); i++)
		{
			auto x = net.forward(testset[i].input);
			auto loss = fish::MSELoss(x, testset[i].output);
			printf("testcase %d/%lu, loss: %f\n", i, testset.size(), loss->data[0]);
		}
	}
}

int main()
{
	srand(seed);
	XorDataset::run();
}