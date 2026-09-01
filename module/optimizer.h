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
	double learn_rate = 1e-3;
	double momentum = 0.0;
	double weight_decay = 0.0;
	std::vector<std::shared_ptr<Tensor>> velocity;

	SGD() {}
	SGD(std::vector<std::shared_ptr<Tensor>> _param, double _learn_rate = 1e-3, double _momentum = 0.0, double _weight_decay = 0.0) : Optimizer(_param), learn_rate(_learn_rate), momentum(_momentum), weight_decay(_weight_decay)
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
	double learn_rate = 1e-3;
	double eps = 1e-8;
	std::vector<std::shared_ptr<Tensor>> sqsum;

	AdaGrad() {}
	AdaGrad(std::vector<std::shared_ptr<Tensor>> _param, double _learn_rate = 1e-3, double _eps = 1e-8) : Optimizer(_param), learn_rate(_learn_rate), eps(_eps)
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
	double learn_rate = 1e-3;
	double beta = 0.9;
	double eps = 1e-8;
	std::vector<std::shared_ptr<Tensor>> average_sqsum;

	RMSProp() {}
	RMSProp(std::vector<std::shared_ptr<Tensor>> _param, double _learn_rate = 1e-3, double _beta = 0.9, double _eps = 1e-8) : Optimizer(_param), learn_rate(_learn_rate), beta(_beta), eps(_eps)
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
	double learn_rate = 1e-3;
	double beta_1 = 0.9;
	double beta_2 = 0.999;
	double weight_decay = 0.01;
	double eps = 1e-8;
	std::vector<std::shared_ptr<Tensor>> average_sum;
	std::vector<std::shared_ptr<Tensor>> average_sqsum;
	int step_count = 0;

	AdamW() {}
	AdamW(std::vector<std::shared_ptr<Tensor>> _param, double _learn_rate = 1e-3, double _beta_1 = 0.9, double _beta_2 = 0.999, double _weight_decay = 0.0, double _eps = 1e-8) : Optimizer(_param), learn_rate(_learn_rate), beta_1(_beta_1), beta_2(_beta_2), weight_decay(_weight_decay), eps(_eps), step_count(0)
	{
		average_sum.resize(_param.size());
		average_sqsum.resize(_param.size());
		for(int i = 0; i < _param.size(); i++)
		{
			average_sum[i] = std::make_shared<Tensor>(Tensor(std::vector<double>(_param[i]->data.size()), _param[i]->shape));
			average_sqsum[i] = std::make_shared<Tensor>(Tensor(std::vector<double>(_param[i]->data.size()), _param[i]->shape));
		}
	}

	void update()
	{
		step_count++;
		for(int i = 0; i < parameters.size(); i++)
		{
			if(parameters[i]->grad == nullptr)
			{
				parameters[i]->grad = std::make_shared<Tensor>(Tensor(std::vector<double>(parameters[i]->data.size()), parameters[i]->shape));
			}

			double bias_correction_1 = 1 - pow(beta_1, step_count);
			double bias_correction_2 = 1 - pow(beta_2, step_count);

			for(int j = 0; j < parameters[i]->data.size(); j++)
			{
				average_sum[i]->data[j] = beta_1 * average_sum[i]->data[j]+ (1 - beta_1) * parameters[i]->grad->data[j];
				average_sqsum[i]->data[j] = beta_2 * average_sqsum[i]->data[j]+ (1 - beta_2) * parameters[i]->grad->data[j] * parameters[i]->grad->data[j];

				double corr_average_sum = average_sum[i]->data[j] / bias_correction_1;
				double corr_average_sqsum = average_sqsum[i]->data[j] / bias_correction_2;

				parameters[i]->data[j] = parameters[i]->data[j] - learn_rate / sqrt(corr_average_sqsum + eps) * corr_average_sum - learn_rate * weight_decay * parameters[i]->data[j];
			}
		}
	}
};

}

#endif