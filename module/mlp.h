/*
多层感知机的封装.
*/

#ifndef _MLP_H
#define _MLP_H

#include <memory>
#include <vector>
#include "module.h"

namespace fish
{

class MLP : BaseNet
{
private:
	SequentialNet net;

public:
	MLP() {}
	MLP(std::vector<size_t> _size)
	{
		std::vector<std::shared_ptr<BaseNet>> layers;
		layers.push_back(std::make_shared<FullyConnectedLayer>(FullyConnectedLayer(_size[0], _size[1])));
		for(int i = 2; i < _size.size(); i++)
		{
			layers.push_back(std::make_shared<ReLULayer>(ReLULayer()));
			layers.push_back(std::make_shared<FullyConnectedLayer>(FullyConnectedLayer(_size[i - 1], _size[i])));
		}
		net = SequentialNet(layers);
	}

	std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) const
	{
		return net.forward(x);
	}

	std::vector<std::shared_ptr<Tensor>> collectParameters()
	{
		return net.collectParameters();
	}
};

}

#endif