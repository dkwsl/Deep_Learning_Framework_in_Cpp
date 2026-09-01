/*
封装激活函数.
*/

#ifndef _LOSS_H
#define _LOSS_H

#include <memory>
#include "autograd.h"

namespace fish
{

// 均方误差（MSE）.
std::shared_ptr<Tensor> MSELoss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target)
{
	auto diff = pred - target;
	auto loss = transpose(diff) * diff;
	return loss;
}

}

#endif