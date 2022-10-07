import pygame
import torch
from networks.encoder import AutoEncoderTrainer, BehaviorAutoEncoder
encoder = BehaviorAutoEncoder()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    trainer = AutoEncoderTrainer(model=encoder, device=device)
    trainer.train()
    trainer.cleanup()
except Exception as e:
    print(e)
    pygame.quit()
