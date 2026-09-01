/*
封装优化器.
*/

#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H

#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>
#include "module.h"

namespace fish
{

class Optimizer
{
public:
	std::vector<std::shared_ptr<Tensor>> parameters;

	Optimizer() = default;
	Optimizer(std::vector<std::shared_ptr<Tensor>> _param)
	{
		parameters = _param;
	}
	virtual ~Optimizer() = default;

	virtual void clearGrad()
	{
		for(int i = 0; i < parameters.size(); i++)
		{
			parameters[i]->grad = nullptr;
		}
	}

	virtual void update() = 0;
};

/*
带动量的随机梯度下降, 默认动量系数为 0.
	-learn_rate: 学习率.
	-momentum: 动量系数.
	-velocity: 每个待优化张量的速度.
*/
class SGD : public Optimizer
{
public:
	double learn_rate = 0.0;
	double momentum = 0.0;
	double weight_decay = 0.0;
	std::vector<std::shared_ptr<Tensor>> velocity;

	SGD() {}
	SGD(std::vector<std::shared_ptr<Tensor>> _param, double _learn_rate, double _momentum = 0.0, double _weight_decay = 0.0) : Optimizer(_param), learn_rate(_learn_rate), momentum(_momentum), weight_decay(_weight_decay)
	{
		velocity.resize(_param.size());
		for(int i = 0; i < _param.size(); i++)
		{
			velocity[i] = std::make_shared<Tensor>(Tensor(std::vector<double>(_param[i]->data.size()), _param[i]->shape));
		}
	}

	void update()
	{
		for(int i = 0; i < parameters.size(); i++)
		{
			if(parameters[i]->grad == nullptr)
			{
				parameters[i]->grad = std::make_shared<Tensor>(Tensor(std::vector<double>(parameters[i]->data.size()), parameters[i]->shape));
			}
			for(int j = 0; j < parameters[i]->data.size(); j++)
			{
				velocity[i]->data[j] = momentum * velocity[i]->data[j] + learn_rate * parameters[i]->grad->data[j];
				parameters[i]->data[j] = parameters[i]->data[j] - velocity[i]->data[j] - learn_rate * weight_decay * parameters[i]->data[j];
			}
		}
	}
};

class AdaGrad : public Optimizer
{
public:
	double learn_rate = 0.0;
	double eps = 1e-8;
	std::vector<std::shared_ptr<Tensor>> sqsum;

	AdaGrad() {}
	AdaGrad(std::vector<std::shared_ptr<Tensor>> _param, double _learn_rate, double _eps = 1e-8) : Optimizer(_param), learn_rate(_learn_rate), eps(_eps)
	{
		sqsum.resize(_param.size());
		for(int i = 0; i < _param.size(); i++)
		{
			sqsum[i] = std::make_shared<Tensor>(Tensor(std::vector<double>(_param[i]->data.size()), _param[i]->shape));
		}
	}

	void update()
	{
		for(int i = 0; i < parameters.size(); i++)
		{
			if(parameters[i]->grad == nullptr)
			{
				parameters[i]->grad = std::make_shared<Tensor>(Tensor(std::vector<double>(parameters[i]->data.size()), parameters[i]->shape));
			}
			for(int j = 0; j < parameters[i]->data.size(); j++)
			{
				sqsum[i]->data[j] += parameters[i]->grad->data[j] * parameters[i]->grad->data[j];
				parameters[i]->data[j] -= learn_rate / sqrt(sqsum[i]->data[j] + eps) * parameters[i]->grad->data[j];
			}
		}
	}
};

class RMSProp : public Optimizer
{
public:
	double learn_rate = 0.0;
	double beta = 0.9;
	double eps = 1e-8;
	std::vector<std::shared_ptr<Tensor>> average_sqsum;

	RMSProp() {}
	RMSProp(std::vector<std::shared_ptr<Tensor>> _param, double _learn_rate, double _beta = 0.9, double _eps = 1e-8) : Optimizer(_param), learn_rate(_learn_rate), beta(_beta), eps(_eps)
	{
		average_sqsum.resize(_param.size());
		for(int i = 0; i < _param.size(); i++)
		{
			average_sqsum[i] = std::make_shared<Tensor>(Tensor(std::vector<double>(_param[i]->data.size()), _param[i]->shape));
		}
	}

	void update()
	{
		for(int i=0;i<parameters.size();i++)
		{
			if(parameters[i]->grad == nullptr)
			{
				parameters[i]->grad = std::make_shared<Tensor>(Tensor(std::vector<double>(parameters[i]->data.size()), parameters[i]->shape));
			}
			for(int j = 0; j < parameters[i]->data.size(); j++)
			{
				average_sqsum[i]->data[j] = beta * average_sqsum[i]->data[j]+ (1 - beta) * parameters[i]->grad->data[j] * parameters[i]->grad->data[j];
				parameters[i]->data[j] -= learn_rate / sqrt(average_sqsum[i]->data[j] + eps) * parameters[i]->grad->data[j];
			}
		}
	}
};

class AdamW : public Optimizer
{
public:
	double learn_rate = 0.0;
	double weight_decay = 0.0;

	AdamW() {}
	AdamW(std::vector<std::shared_ptr<Tensor>> _param, double _learn_rate, double _weight_decay = 0.0) : Optimizer(_param), learn_rate(_learn_rate), weight_decay(_weight_decay) {}

	void update()
	{
		;
	}
};

}

#endif