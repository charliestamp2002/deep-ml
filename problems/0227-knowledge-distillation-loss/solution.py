import torch
import torch.nn.functional as F

def distillation_loss(
	student_logits: torch.Tensor,
	teacher_logits: torch.Tensor,
	temperature: float = 1.0
) -> torch.Tensor:
	"""
	Compute knowledge distillation loss using PyTorch.
	
	Args:
		student_logits: Logits from student model
		teacher_logits: Logits from teacher model
		temperature: Softmax temperature
		
	Returns:
		Distillation loss
	"""
	teacher_scaled = teacher_logits/temperature
	student_scaled = student_logits/temperature

	T_probs = torch.softmax(teacher_scaled,dim=-1)
	S_probs = torch.softmax(student_scaled,dim=-1)

	kl = T_probs * torch.log(T_probs/S_probs)

	loss = (temperature ** 2) * torch.sum(kl)
	return loss



	