/*
定义神经网络网络基类与神经网络的一些基本组件, 包括顺序连接网络类，全连接层和激活函数 ReLU 层.
*/

#ifndef _MODULE_H
#define _MODULE_H

#include <random>
#include <memory>
#include <vector>
#include "autograd.h"

namespace fish
{

const unsigned int seed = 114514;
std::mt19937 gen(seed);
std::normal_distribution<double> rand_normal(0, 0.1);

/*
BaseNet: 所有神经网络的基类.
	-parameters: 需要优化的参数.
	-forward: 前向传播函数.
	-collectParameters: 收集需要训练的参数.
*/
class BaseNet
{
public:
	std::vector<std::shared_ptr<Tensor>> parameters;
	BaseNet() {}
	virtual ~BaseNet() = default;

	virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) const = 0;
	virtual std::vector<std::shared_ptr<Tensor>> collectParameters()
	{
		return {};
	}
};

/*
SequentialNet: 将多个网络按顺序连接成整体.
	-layers: 按顺序存储指向每个网络的指针.
	-forward: 前向传播函数, 前一层的输出作为当前层的输入.
	-collectParameters: 依次收集需要训练的参数.
*/
class SequentialNet : public BaseNet
{
public:
	std::vector<std::shared_ptr<BaseNet>> layers;

	SequentialNet() {}
	SequentialNet(std::vector<std::shared_ptr<BaseNet>> _layers) : layers(_layers) {}
	~SequentialNet() = default;

	std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) const
	{
		for(int i = 0; i < layers.size(); i++)
		{
			x = layers[i]->forward(x);
		}
		return x;
	}

	std::vector<std::shared_ptr<Tensor>> collectParameters()
	{
		std::vector<std::shared_ptr<Tensor>> param;
		for(int i = 0; i < layers.size(); i++)
		{
			auto layer_param = layers[i]->collectParameters();
			param.reserve(param.size() + layer_param.size());
			param.insert(param.end(), layer_param.begin(), layer_param.end());
		}
		return param;
	}
};

/*
FullyConnectedLayer: 全连接层, 输出向量为输入的线性组合.
	-weight: 输入向量对输出向量每个位置的影响权重.
	-bias: 输出向量的偏置.
	-forward: 输入向量 x, 返回线性组合 weight * x + bias.
	-collectParameters: 返回需要训练的参数 weight 和 bias.
*/
class FullyConnectedLayer : public BaseNet
{
protected:
	std::shared_ptr<Tensor> weight, bias;
public:
	FullyConnectedLayer() {}
	FullyConnectedLayer(size_t _input, size_t _output)
	{
		weight = std::make_shared<Tensor>(Tensor(std::vector<double>(_output * _input), {_output, _input}, true));
		bias = std::make_shared<Tensor>(Tensor(std::vector<double>(_output), {_output, 1}, true));

		for(int i = 0; i < weight->data.size(); i++)
		{
			weight->data[i] = rand_normal(gen);
		}
		for(int i = 0; i < bias->data.size(); i++)
		{
			bias->data[i] = rand_normal(gen);
		}
	}
	~FullyConnectedLayer() = default;

	std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) const
	{
		return weight * x + bias;
	}

	std::vector<std::shared_ptr<Tensor>> collectParameters()
	{
		return {weight, bias};
	}
};

/*
ReLULayer: 激活函数 ReLU 层.
	-forward: 对输入的每个元素作用 ReLU.
*/
class ReLULayer : public BaseNet
{
public:
	ReLULayer() {}
	~ReLULayer() = default;

	std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) const
	{
		return ReLU(x);
	}
};

}

#endif