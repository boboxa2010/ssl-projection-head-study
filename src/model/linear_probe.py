import torch
from torch import nn
from src.model.proj_clf import ProjectedClassifier
from src.model.sim_clr import SimCLR

class LinearProbe(nn.Module):
    def __init__(self, model_type = 'ProjectedClassifier', mode = 'pre', num_classes = 20, head_mode = 'mlp'):
        super(LinearProbe, self).__init__()
        """
        Universal model for linear probing.
        model = pretrained model (SimCLR or ProjectedClassifier)
        num_classes: number of classes for CrossEntropy
        two modes: 
        'pre' if pre-projections outputs (features)
        'post' if post-projections outputs (projections)
        """
        # Прогоняем одну картинку, чтобы узнать размер (512? 128? 768?)
        dummy_input = torch.randn(1, 3, 32, 32)
        self.model = ProjectedClassifier(mode=head_mode) if model_type == "ProjectedClassifier" else SimCLR(mode=head_mode, kappa=2.28) # пока так
        # потом подгрузится чекпоинт
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        with torch.no_grad():
            dummy_out = self.model(dummy_input)
            if mode == 'pre':
                feat_dim = dummy_out['features'].shape[1]
            elif mode == 'post':
                feat_dim = dummy_out['projections'].shape[1]
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
        self.mode = mode
    
        self.probe_classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, img: torch.Tensor, **batch) -> dict:

        self.model.eval()

        with torch.no_grad():
            model_output = self.model(img)
            if self.mode == 'pre':
                reps = model_output['features']
            elif self.mode == 'post':
                reps = model_output['projections']
        
        return {
            "logits": self.probe_classifier(reps)
        }
        
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info