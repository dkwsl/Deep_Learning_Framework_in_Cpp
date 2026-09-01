#include <cstdio>
#include <cstdlib>
#include <memory>
#include "module/module.h"
#include "module/loss.h"
#include "module/optimizer.h"

const unsigned int seed = 114514;
const double eps = 1e-4;

class MLP : public fish::BaseNet
{
public:
	fish::SequentialNet net;

	MLP() {}
	MLP(size_t _input_size, size_t _hidden_size, size_t _output_size)
	{
		net = fish::SequentialNet({std::make_shared<fish::FullyConnectedLayer>(fish::FullyConnectedLayer(_input_size, _hidden_size)),
		                           std::make_shared<fish::ReLULayer>(fish::ReLULayer()),
						   std::make_shared<fish::FullyConnectedLayer>(fish::FullyConnectedLayer(_hidden_size, _output_size))});
	}
	MLP(size_t _input_size, size_t _hidden_size_1, size_t _hidden_size_2, size_t _output_size)
	{
		net = fish::SequentialNet({std::make_shared<fish::FullyConnectedLayer>(fish::FullyConnectedLayer(_input_size, _hidden_size_1)),
		                           std::make_shared<fish::ReLULayer>(fish::ReLULayer()),
						   std::make_shared<fish::FullyConnectedLayer>(fish::FullyConnectedLayer(_hidden_size_1, _hidden_size_2)),
						   std::make_shared<fish::ReLULayer>(fish::ReLULayer()),
						   std::make_shared<fish::FullyConnectedLayer>(fish::FullyConnectedLayer(_hidden_size_2, _output_size))});
	}

	std::shared_ptr<fish::Tensor> forward(std::shared_ptr<fish::Tensor> x) const
	{
		return net.forward(x);
	}

	std::vector<std::shared_ptr<fish::Tensor>> collectParameters()
	{
		return net.collectParameters();
	}
};

MLP net;
fish::SGD optim;

namespace XorDataset
{
	const int epoch = 1000000;
	const double learn_rate = 0.0001;
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
		net = MLP(2, 64, 64, 1);
		optim = fish::SGD(net.collectParameters(), learn_rate, momentum, weight_decay);
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