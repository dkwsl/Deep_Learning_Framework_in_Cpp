/*
张量和计算图. 记录前向传播时张量的计算图, 用 Tensor::need_grad 参数记录张量是否应该进入计算图,
张量带一个指针指向算出这个张量的计算图节点, 计算图节点带一列指针指向这个计算需要的张量. 存储图的
顺序与计算顺序相反.

tensor_a <-----|
            AddNode <----- tensor_c = tensor_a + tensor_b
tensor_b <-----|

tensor_a <----- ReLUNode <----- tensor_b = ReLU(tensor_a)

反向传播时从唯一的终点, 也即 loss 节点, 按照拓扑序转移.
*/

#ifndef _AUTOGRAD_H
#define _AUTOGRAD_H

#include <cstddef>
#include <memory>
#include <vector>

namespace fish
{

// 是否启动计算图.
bool enable_autograd = true;
// 启用计算图与自动微分.
void enableAutograd()
{
	enable_autograd = true;
}
// 停用计算图与自动微分, 之后的操作不再新建计算图节点, 但之前的计算图不会被清除.
void disableAutograd()
{
	enable_autograd = false;
}


class Tensor;
class Node;

/*
Tensor: 张量类.
	-data: 扁平化存储张量的全部元素, 靠后的维度内存连续.
	-shape: 每一维的大小.
	-stride: 每一维在 data 上的步长.
	-grad: 梯度.
	-node: 对应的计算图节点指针.
	-need_grad: 当前点是否需要记录梯度, 以此判断是否需要加入计算图.
	-topo_vis: 拓扑排序是否已经经过当前点.
*/
class Tensor
{
public:
	std::vector<double> data;
	std::vector<size_t> shape;
	std::vector<size_t> stride;
	std::shared_ptr<Tensor> grad;
	std::shared_ptr<Node> node;
	bool need_grad = false;
	bool topo_vis = false;

	Tensor() {}
	Tensor(const std::vector<double> &_data, const std::vector<size_t> &_shape) : data(_data), shape(_shape)
	{
		getStride();
		grad = nullptr;
		node = nullptr;
		topo_vis = false;
	}
	Tensor(const std::vector<double> &_data, const std::vector<size_t> &_shape, bool _need_grad) : data(_data), shape(_shape), need_grad(_need_grad)
	{
		getStride();
		node = nullptr;
		topo_vis = false;

		if(_need_grad)
		{
			grad = std::make_shared<Tensor>(Tensor(std::vector<double>(_data.size()), _shape));
		}
		else
		{
			grad = nullptr;
		}
	}

	// 计算每一维的步长, 即 shape 的后缀积.
	void getStride()
	{
		if(shape.empty()) return;
		stride.resize(shape.size());
		stride[(int)stride.size() - 1] = 1;
		for(int i = (int)stride.size() - 2; i >= 0; i--)
		{
			stride[i] = stride[i + 1] * shape[i + 1];
		}
	}
	// 多维下标映射到 data 下标, 下标从 0 开始.
	size_t index(std::vector<size_t> _index)
	{
		size_t ind = 0;
		for(int i = 0; i < _index.size(); i++)
		{
			ind += _index[i] * stride[i];
		}
		return ind;
	}
};

/*
Node: 计算图节点类.
	-inputs: 所有指向输入张量的指针.
*/
class Node
{
public:
	std::vector<std::shared_ptr<Tensor>> inputs;

	Node() {}
	virtual ~Node() = default;

	// 反向传播.
	virtual void backward(std::shared_ptr<Tensor> back_grad) = 0;
};

std::shared_ptr<Tensor> operator + (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> operator - (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> operator * (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> transpose(std::shared_ptr<Tensor> a);

// 加法计算图节点.
class AddNode : public Node
{
public:
	AddNode() {}
	AddNode(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b)
	{
		inputs = {a, b};
	}

	void backward(std::shared_ptr<Tensor> back_grad)
	{
		if(inputs[0]->need_grad)
		{
			if(inputs[0]->grad == nullptr)
			{
				inputs[0]->grad = std::make_shared<Tensor>(Tensor(std::vector<double>(inputs[0]->data.size()), inputs[0]->shape));
			}
			inputs[0]->grad = inputs[0]->grad + back_grad;
		}
		if(inputs[1]->need_grad)
		{
			if(inputs[1]->grad == nullptr)
			{
				inputs[1]->grad = std::make_shared<Tensor>(Tensor(std::vector<double>(inputs[1]->data.size()), inputs[1]->shape));
			}
			inputs[1]->grad = inputs[1]->grad + back_grad;
		}
	}
};

// 减法计算图节点.
class SubNode : public Node
{
public:
	SubNode() {}
	SubNode(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b)
	{
		inputs = {a, b};
	}
	
	void backward(std::shared_ptr<Tensor> back_grad)
	{
		if(inputs[0]->need_grad)
		{
			if(inputs[0]->grad == nullptr)
			{
				inputs[0]->grad = std::make_shared<Tensor>(Tensor(std::vector<double>(inputs[0]->data.size()), inputs[0]->shape));
			}
			inputs[0]->grad = inputs[0]->grad + back_grad;
		}
		if(inputs[1]->need_grad)
		{
			if(inputs[1]->grad == nullptr)
			{
				inputs[1]->grad = std::make_shared<Tensor>(Tensor(std::vector<double>(inputs[1]->data.size()), inputs[1]->shape));
			}
			inputs[1]->grad = inputs[1]->grad - back_grad;
		}
	}
};

// 矩阵乘法计算图节点.
class MatmulNode : public Node
{
public:
	MatmulNode() {}
	MatmulNode(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b)
	{
		inputs = {a, b};
	}

	void backward(std::shared_ptr<Tensor> back_grad)
	{
		if(inputs[0]->need_grad)
		{
			if(inputs[0]->grad == nullptr)
			{
				inputs[0]->grad = std::make_shared<Tensor>(Tensor(std::vector<double>(inputs[0]->data.size()), inputs[0]->shape));
			}
			inputs[0]->grad = inputs[0]->grad + back_grad * transpose(inputs[1]);
		}
		if(inputs[1]->need_grad)
		{
			if(inputs[1]->grad == nullptr)
			{
				inputs[1]->grad = std::make_shared<Tensor>(Tensor(std::vector<double>(inputs[1]->data.size()), inputs[1]->shape));
			}
			inputs[1]->grad = inputs[1]->grad + transpose(inputs[0]) * back_grad;
		}
	}
};

// 转置计算图节点.
class TransposeNode : public Node
{
public:
	TransposeNode() {}
	TransposeNode(std::shared_ptr<Tensor> a)
	{
		inputs = {a};
	}

	void backward(std::shared_ptr<Tensor> back_grad)
	{
		if(inputs[0]->need_grad)
		{
			if(inputs[0]->grad == nullptr)
			{
				inputs[0]->grad = std::make_shared<Tensor>(Tensor(std::vector<double>(inputs[0]->data.size()), inputs[0]->shape));
			}
			inputs[0]->grad = inputs[0]->grad + transpose(back_grad);
		}
	}
};

// ReLU 函数计算图节点.
class ReLUNode : public Node
{
public:
	ReLUNode() {}
	ReLUNode(std::shared_ptr<Tensor> a)
	{
		inputs = {a};
	}

	void backward(std::shared_ptr<Tensor> back_grad)
	{
		if(inputs[0]->need_grad)
		{
			if(inputs[0]->grad == nullptr)
			{
				inputs[0]->grad = std::make_shared<Tensor>(Tensor(std::vector<double>(inputs[0]->data.size()), inputs[0]->shape));
			}
			for(int i=0;i<inputs[0]->data.size();i++)
			{
				if(inputs[0]->data[i] > 0)
				{
					inputs[0]->grad->data[i] = inputs[0]->grad->data[i] + back_grad->data[i];
				}
			}
		}
	}
};

// 张量加法, 同时生成计算图, 要求 a, b 两张量 shape 相同.
std::shared_ptr<Tensor> operator + (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b)
{
	std::vector<double> res_data(a->data.size());
	for(int i = 0; i < a->data.size(); i++)
	{
		res_data[i] = a->data[i] + b->data[i];
	}

	auto res = std::make_shared<Tensor>(Tensor(res_data, a->shape, a->need_grad || b->need_grad));

	if(enable_autograd && res->need_grad)
	{
		res->node = std::make_shared<AddNode>(AddNode(a, b));
	}

	return res;
}

// 张量减法, 同时生成计算图, 要求 a, b 两张量 shape 相同.
std::shared_ptr<Tensor> operator - (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b)
{
	std::vector<double> res_data(a->data.size());
	for(int i = 0; i < a->data.size(); i++)
	{
		res_data[i] = a->data[i] - b->data[i];
	}

	auto res = std::make_shared<Tensor>(Tensor(res_data, a->shape, a->need_grad || b->need_grad));

	if(enable_autograd && res->need_grad)
	{
		res->node = std::make_shared<SubNode>(SubNode(a, b));
	}

	return res;
}

// 矩阵乘法, 同时生成计算图, 要求 a, b 均为矩阵, 且 a 的列数 a->shape[1] 与 b 的行数 b->shape[0] 相等.
std::shared_ptr<Tensor> operator * (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b)
{
	size_t n = a->shape[0], m = b->shape[1], k = a->shape[1];
	std::vector<double> res_data(n * m);
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < m; j++)
		{
			double res = 0;
			for(int l = 0; l < k; l++)
			{
				res += a->data[i * k + l] * b->data[l * m + j];
			}
			res_data[i * m + j] = res;
		}
	}

	auto res = std::make_shared<Tensor>(Tensor(res_data, std::vector<size_t>{n, m}, a->need_grad || b->need_grad));

	if(enable_autograd && res->need_grad)
	{
		res->node = std::make_shared<MatmulNode>(MatmulNode(a, b));
	}

	return res;
}

// 矩阵转置.
std::shared_ptr<Tensor> transpose(std::shared_ptr<Tensor> a)
{
	size_t n = a->shape[0], m = a->shape[1];
	std::vector<double> res_data(n * m);
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < m; j++)
		{
			res_data[j * n + i] = a->data[i * m + j];
		}
	}

	auto res = std::make_shared<Tensor>(Tensor(res_data, std::vector<size_t>{m, n}, a->need_grad));

	if(enable_autograd && res->need_grad)
	{
		res->node = std::make_shared<TransposeNode>(TransposeNode(a));
	}

	return res;
}

inline double ReLU(double x)
{
	return x > 0 ? x : 0;
}
// 对矩阵全部函数作用 ReLU.
std::shared_ptr<Tensor> ReLU(std::shared_ptr<Tensor> a)
{
	std::vector<double> res_data(a->data.size());
	for(int i = 0; i < a->data.size(); i++)
	{
		res_data[i] = ReLU(a->data[i]);
	}

	auto res = std::make_shared<Tensor>(Tensor(res_data, a->shape, a->need_grad));

	if(enable_autograd && res->need_grad)
	{
		res->node = std::make_shared<ReLUNode>(ReLUNode(a));
	}

	return res;
}

// 计算图拓扑排序.
void topoSort(std::shared_ptr<Tensor> now_tensor, std::vector<std::shared_ptr<Tensor>> &topo_order)
{
	if(now_tensor->topo_vis) return;
	now_tensor->topo_vis = true;

	auto now_node = now_tensor->node;
	if(now_node != nullptr)
	{
		for(int i=0;i<now_node->inputs.size();i++)
		{
			topoSort(now_node->inputs[i], topo_order);
		}
	}

	topo_order.emplace_back(now_tensor);
}

// 反向传播.
void backward(std::shared_ptr<Tensor> root)
{
	disableAutograd(); 

	std::vector<std::shared_ptr<Tensor>> topo_order;
	topoSort(root, topo_order);

	root->grad = std::make_shared<Tensor>(Tensor({1}, {1, 1}));
	for(auto it = topo_order.rbegin(); it != topo_order.rend(); it++)
	{
		if((*it)->node != nullptr)
		{
			(*it)->node->backward((*it)->grad);
		}
	}

	for(auto it = topo_order.begin(); it != topo_order.end(); it++)
	{
		(*it)->node = nullptr;
		(*it)->topo_vis = false;
	}

	enableAutograd();
}

}

#endif