import torch
from torch import nn
from src.model.proj_clf import ProjectedClassifier
from src.model.sim_clr import SimCLR

class LinearProbe(nn.Module):
    def __init__(self, model_type = 'ProjectedClassifier', mode = 'pre', num_classes = 20, head_mode = 'mlp', pretrain_type = 'scl'):
        super(LinearProbe, self).__init__()
        """
        Universal model for linear probing.
        model = pretrained model (SimCLR or ProjectedClassifier)
        num_classes: number of classes for CrossEntropy
        two modes: 
        'pre' if pre-projections outputs (features)
        'post' if post-projections outputs (projections)
        pretrain_type: scl, ssl, sl 
        """
        # params are not important because wil be loaded from checkpoint - no learn from scratch
        if pretrain_type == 'scl':
            self.model = ProjectedClassifier(mode=head_mode, kappa=2.28)
        elif pretrain_type == 'sl':
            self.model = ProjectedClassifier(n_class=20, mode=head_mode, kappa=2.28) # pretrained on coarse everythhing
        elif pretrain_type == 'ssl':
            self.model = SimCLR(mode=head_mode, kappa=2.28)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        # dummy image to know the output_dim (512? 128? 768?)
        dummy_input = torch.randn(1, 3, 32, 32)

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