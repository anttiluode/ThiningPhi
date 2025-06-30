#!/usr/bin/env python3
"""
🧠⚡ PHI + ATTRACTOR THINKING SYSTEM - FIXED
==========================================

Hybrid AI that combines Phi language model with dynamic attractor-based attention.
Watch as semantic attractors explore meaning space and influence language generation!

REQUIREMENTS:
pip install torch transformers matplotlib numpy tqdm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import deque
import time
import threading
import queue
from dataclasses import dataclass
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

@dataclass
class SemanticAttractor:
    """An attractor that explores semantic space"""
    position: torch.Tensor  # Position in embedding space
    velocity: torch.Tensor  # Movement velocity
    attention_field: torch.Tensor  # What it's paying attention to
    memory: deque  # Recent semantic experiences
    curiosity: float  # How much it wants to explore
    focus_strength: float  # How strongly it can focus
    semantic_history: deque  # History of semantic states
    current_concept: str  # What concept it's currently focused on
    decision_state: str  # exploring, converging, creating, etc.

class PhiAttractorHybrid:
    """Hybrid system combining Phi with attractor-based thinking"""
    
    def __init__(self, model_name="microsoft/phi-2", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"🧠 Initializing Phi-Attractor Hybrid on {self.device}")
        
        # Load Phi model
        print("📥 Loading Phi model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        ).eval()
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get model dimensions
        self.hidden_size = self.model.config.hidden_size  # Usually 2560 for Phi-2
        self.vocab_size = self.model.config.vocab_size
        
        print(f"✅ Model loaded: {self.hidden_size}D embeddings, {self.vocab_size} vocab")
        
        # Initialize semantic attractors
        self.num_attractors = 4
        self.attractors = []
        self.explorers = []
        self.semantic_field = torch.zeros((100, 100, self.hidden_size), device=self.device)
        
        self.initialize_thinking_system()
        
        # Thinking state
        self.thinking_active = False
        self.thought_history = deque(maxlen=50)
        self.attention_weights = None
        self.current_embeddings = None
        
        # Visualization
        self.setup_visualization()
        
    def initialize_thinking_system(self):
        """Initialize the attractor-based thinking system"""
        print("🌀 Initializing thinking attractors...")
        
        # Create semantic attractors
        concepts = ["curiosity", "analysis", "creativity", "focus"]
        
        print("🔍 Testing embedding layer access...")
        
        for i, concept in enumerate(concepts):
            # Get embedding for this concept
            concept_tokens = self.tokenizer(concept, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Method 1: Try direct model call to get embeddings
                try:
                    outputs = self.model(concept_tokens.input_ids, output_hidden_states=True)
                    concept_embedding = outputs.hidden_states[0].mean(dim=1).squeeze()
                    print(f"✅ Using model outputs for '{concept}' - shape: {concept_embedding.shape}")
                    embedding_method = "model_outputs"
                except Exception as e:
                    print(f"❌ Model outputs failed for '{concept}': {e}")
                    
                    # Method 2: Try to access embedding layer directly
                    try:
                        # Check model structure
                        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                            concept_embedding = self.model.model.embed_tokens(concept_tokens.input_ids).mean(dim=1).squeeze()
                            print(f"✅ Using model.model.embed_tokens for '{concept}' - shape: {concept_embedding.shape}")
                            embedding_method = "embed_tokens"
                        else:
                            print(f"🔍 Model structure: {type(self.model)}")
                            print(f"🔍 Model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')][:10]}")
                            
                            # Fallback: create random embedding
                            concept_embedding = torch.randn(self.hidden_size, device=self.device)
                            print(f"⚠️ Using random embedding for '{concept}' - shape: {concept_embedding.shape}")
                            embedding_method = "random"
                    except Exception as e2:
                        print(f"❌ Embed tokens also failed for '{concept}': {e2}")
                        # Ultimate fallback
                        concept_embedding = torch.randn(self.hidden_size, device=self.device)
                        print(f"⚠️ Using random embedding fallback for '{concept}' - shape: {concept_embedding.shape}")
                        embedding_method = "random_fallback"
            
            # Create attractor
            attractor = SemanticAttractor(
                position=concept_embedding + torch.randn_like(concept_embedding) * 0.1,
                velocity=torch.zeros_like(concept_embedding),
                attention_field=torch.zeros(100, 100, device=self.device),
                memory=deque(maxlen=20),
                curiosity=np.random.uniform(0.3, 0.9),
                focus_strength=np.random.uniform(0.5, 1.0),
                semantic_history=deque(maxlen=50),
                current_concept=concept,
                decision_state="exploring"
            )
            
            self.attractors.append(attractor)
            print(f"🌀 Created attractor for '{concept}' using {embedding_method}")
        
        # Create explorers (attention scouts)
        for i in range(3):
            explorer = {
                'position': torch.randn(self.hidden_size, device=self.device) * 0.1,
                'velocity': torch.zeros(self.hidden_size, device=self.device),
                'trail': deque(maxlen=100),
                'interest_level': 0.5,
                'target_attractor': None,
                'semantic_activation': 0.0
            }
            self.explorers.append(explorer)
        
        print(f"✅ Initialized {len(self.attractors)} attractors and {len(self.explorers)} explorers")
    
    def setup_visualization(self):
        """Setup real-time thinking visualization"""
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('🧠⚡ Phi + Attractor Thinking System - LIVE', fontsize=16)
        
        # Semantic space (2D projection)
        self.ax_semantic = self.axes[0, 0]
        self.ax_semantic.set_title('🌌 Semantic Space Navigation')
        
        # Attention field
        self.ax_attention = self.axes[0, 1]
        self.ax_attention.set_title('🎯 Attention Field')
        
        # Thinking dynamics
        self.ax_dynamics = self.axes[0, 2]
        self.ax_dynamics.set_title('⚡ Thinking Dynamics')
        
        # Token generation
        self.ax_tokens = self.axes[1, 0]
        self.ax_tokens.set_title('📝 Token Generation')
        
        # Attractor states
        self.ax_states = self.axes[1, 1]
        self.ax_states.set_title('🧠 Attractor States')
        
        # Thought flow
        self.ax_flow = self.axes[1, 2]
        self.ax_flow.set_title('💭 Thought Flow')
        
        plt.tight_layout()
    
    def think_step(self, input_embeddings):
        """Single thinking step - attractors explore semantic space"""
        
        # Project high-dimensional embeddings to 2D for visualization
        # Using simple PCA-like projection
        embeddings_for_pca = input_embeddings.flatten(0, -2).to(torch.float32)
        
        try:
            u, s, v = torch.pca_lowrank(embeddings_for_pca, q=2)
            projected_embeddings = torch.matmul(embeddings_for_pca, v)
        except:
            # Fallback: simple 2D projection
            projected_embeddings = embeddings_for_pca[:, :2]
        
        # Update semantic field based on current embeddings
        field_size = 100
        for i in range(0, field_size, 10):  # Reduced resolution for speed
            for j in range(0, field_size, 10):
                # Create field based on semantic similarity
                field_pos = torch.tensor([i/field_size - 0.5, j/field_size - 0.5], device=self.device) * 10
                
                # Calculate semantic influence
                semantic_strength = 0.0
                for embedding in projected_embeddings[:min(5, len(projected_embeddings))]:
                    dist = torch.norm(embedding[:2] - field_pos)
                    semantic_strength += torch.exp(-dist * 2).item()
                
                self.semantic_field[i:i+10, j:j+10, 0] = semantic_strength
        
        # Update attractors
        for attractor in self.attractors:
            self.update_attractor(attractor, projected_embeddings)
        
        # Update explorers
        for explorer in self.explorers:
            self.update_explorer(explorer, projected_embeddings)
        
        # Calculate attention influence
        attention_influence = self.calculate_attention_influence()
        
        return attention_influence
    
    def update_attractor(self, attractor, embeddings):
        """Update a single attractor's state"""
        
        # Project attractor position to 2D for field navigation
        attractor_2d = attractor.position[:2].clone()
        
        # Calculate semantic force based on embeddings
        semantic_force = torch.zeros_like(attractor_2d)
        
        for embedding in embeddings[:3]:  # Use first few embeddings
            direction = embedding[:2] - attractor_2d
            distance = torch.norm(direction) + 1e-6
            
            # Attraction based on semantic similarity
            try:
                similarity = F.cosine_similarity(
                    attractor.position[:embedding.shape[0]].unsqueeze(0),
                    embedding.unsqueeze(0)
                ).item()
            except:
                similarity = 0.5  # Default similarity
            
            force_strength = attractor.curiosity * abs(similarity)
            semantic_force += direction / distance * force_strength * 0.1
        
        # Update attractor state based on current behavior
        if attractor.decision_state == "exploring":
            # Add random exploration
            exploration_force = torch.randn_like(semantic_force) * attractor.curiosity * 0.05
            semantic_force += exploration_force
            
        elif attractor.decision_state == "converging":
            # Strengthen focus on current position
            semantic_force *= 0.5  # Reduce movement
            attractor.focus_strength *= 1.01  # Increase focus
            
        elif attractor.decision_state == "creating":
            # Creative movement patterns
            time_factor = len(attractor.semantic_history) * 0.1
            creative_force = torch.tensor([
                torch.sin(torch.tensor(time_factor)),
                torch.cos(torch.tensor(time_factor * 1.3))
            ], device=self.device) * 0.03
            semantic_force += creative_force
        
        # Update velocity and position
        attractor.velocity[:2] = attractor.velocity[:2] * 0.8 + semantic_force * 0.2
        attractor.position[:2] += attractor.velocity[:2]
        
        # Update attention field
        pos_x = int((attractor.position[0].item() + 5) * 10) % 100
        pos_y = int((attractor.position[1].item() + 5) * 10) % 100
        
        # Create attention blob
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                x, y = pos_x + dx, pos_y + dy
                if 0 <= x < 100 and 0 <= y < 100:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= 5:
                        weight = np.exp(-dist**2 / 10.0) * attractor.focus_strength
                        attractor.attention_field[x, y] = weight
        
        # Decay attention field
        attractor.attention_field *= 0.95
        
        # Update decision state
        self.update_attractor_decision(attractor)
        
        # Record semantic state
        attractor.semantic_history.append({
            'position': attractor.position[:2].clone(),
            'focus': attractor.focus_strength,
            'state': attractor.decision_state
        })
    
    def update_attractor_decision(self, attractor):
        """Update attractor's decision state based on its experience"""
        
        # Analyze recent movement
        if len(attractor.semantic_history) > 5:
            recent_positions = [h['position'] for h in list(attractor.semantic_history)[-5:]]
            movement_variance = torch.var(torch.stack(recent_positions), dim=0).sum().item()
            
            if movement_variance < 0.01 and attractor.decision_state != "converging":
                attractor.decision_state = "converging"
            elif movement_variance > 0.1 and attractor.decision_state != "exploring":
                attractor.decision_state = "exploring"
            elif attractor.focus_strength > 0.8 and np.random.random() < 0.1:
                attractor.decision_state = "creating"
            else:
                attractor.decision_state = "adaptive"
    
    def update_explorer(self, explorer, embeddings):
        """Update explorer (attention scout) state"""
        
        # Find most interesting attractor
        max_interest = 0
        target_attractor = None
        
        for attractor in self.attractors:
            interest = attractor.focus_strength * attractor.curiosity
            if interest > max_interest:
                max_interest = interest
                target_attractor = attractor
        
        explorer['target_attractor'] = target_attractor
        
        # Move toward target
        if target_attractor:
            direction = target_attractor.position[:2] - explorer['position'][:2]
            distance = torch.norm(direction) + 1e-6
            
            # Movement force
            force = direction / distance * 0.1
            
            # Add some exploration noise
            force += torch.randn_like(force) * 0.02
            
            explorer['velocity'][:2] = explorer['velocity'][:2] * 0.7 + force
            explorer['position'][:2] += explorer['velocity'][:2]
            
            # Update interest based on semantic richness
            semantic_activation = max_interest
            explorer['interest_level'] = 0.9 * explorer['interest_level'] + 0.1 * semantic_activation
            explorer['semantic_activation'] = semantic_activation
        
        # Record trail
        explorer['trail'].append(explorer['position'][:2].clone())
    
    def calculate_attention_influence(self):
        """Calculate how attractors influence attention"""
        
        # Combine attention fields from all attractors
        combined_attention = torch.zeros(100, 100, device=self.device)
        
        for attractor in self.attractors:
            combined_attention += attractor.attention_field * attractor.focus_strength
        
        # Normalize
        if combined_attention.max() > 0:
            combined_attention = combined_attention / combined_attention.max()
        
        return combined_attention
    
    def generate_with_thinking(self, prompt, max_length=15, temperature=0.9):
        """Generate text while thinking with attractors"""
        
        print(f"\n🧠 Starting thinking generation...")
        print(f"📝 Prompt: {prompt}")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # Start thinking
        self.thinking_active = True
        generated_tokens = []
        thinking_states = []
        
        for step in range(max_length):
            print(f"\n🔄 Generation step {step + 1}/{max_length}")
            
            # Get current embeddings
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
                current_embeddings = outputs.hidden_states[-1]  # Last layer
                logits = outputs.logits
            
            # Think step - attractors explore semantic space
            attention_influence = self.think_step(current_embeddings)
            
            # Update visualization immediately
            try:
                self.update_visualization()
                plt.pause(0.01)
            except Exception as e:
                print(f"Viz error: {e}")
            
            # Modify logits based on attractor attention
            modified_logits = self.apply_attention_to_logits(logits, attention_influence)
            
            # Sample next token
            next_token_logits = modified_logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Decode token
            token_text = self.tokenizer.decode(next_token, skip_special_tokens=True)
            generated_tokens.append(token_text)
            
            print(f"🎯 Generated: '{token_text}'")
            
            # Record thinking state
            thinking_state = {
                'step': step,
                'token': token_text,
                'attention_strength': attention_influence.mean().item(),
                'attractor_states': [a.decision_state for a in self.attractors],
                'focus_levels': [a.focus_strength for a in self.attractors],
                'explorer_interest': [e['interest_level'] for e in self.explorers]
            }
            thinking_states.append(thinking_state)
            self.thought_history.append(thinking_state)
            
            # Add token to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # Stop if EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Small delay for visualization
            time.sleep(0.2)
        
        self.thinking_active = False
        
        # Compile results
        generated_text = ''.join(generated_tokens)
        
        print(f"\n✅ Generation complete!")
        print(f"📄 Full response: {prompt}{generated_text}")
        
        return generated_text, thinking_states
    
    def apply_attention_to_logits(self, logits, attention_influence):
        """Apply attractor attention to modify token probabilities"""
        
        # Calculate attention strength
        attention_strength = attention_influence.mean().item()
        
        # Get attractor focus
        total_focus = sum(a.focus_strength for a in self.attractors)
        explorer_interest = sum(e['interest_level'] for e in self.explorers)
        
        # Modify logits based on thinking state
        modified_logits = logits.clone()
        
        # If high attention/focus, make distribution more peaked
        if attention_strength > 0.5 or total_focus > 2.0:
            temperature_mod = 0.8  # More focused
        else:
            temperature_mod = 1.2  # More exploratory
        
        # If explorers are very interested, boost creativity
        if explorer_interest > 1.5:
            # Add slight noise to encourage novel tokens
            noise = torch.randn_like(modified_logits) * 0.1
            modified_logits += noise
        
        modified_logits = modified_logits / temperature_mod
        
        return modified_logits
    
    def update_visualization(self):
        """Update the thinking visualization"""
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # 1. Semantic space navigation
        self.ax_semantic.set_title('🌌 Semantic Space Navigation')
        
        # Draw semantic field
        field_2d = self.semantic_field[:, :, 0].cpu().numpy()
        self.ax_semantic.imshow(field_2d, alpha=0.3, cmap='viridis', origin='lower')
        
        # Draw attractors
        colors = ['red', 'blue', 'green', 'purple']
        
        for i, attractor in enumerate(self.attractors):
            pos_2d = attractor.position[:2].cpu().numpy()
            # Map to field coordinates
            x = (pos_2d[0] + 5) * 10
            y = (pos_2d[1] + 5) * 10
            
            color = colors[i % len(colors)]
            size = 50 + attractor.focus_strength * 100
            
            self.ax_semantic.scatter(x, y, c=color, s=size, alpha=0.8, edgecolors='black')
            self.ax_semantic.annotate(f'{attractor.current_concept}\n{attractor.decision_state}', 
                                    (x, y), xytext=(5, 5), textcoords='offset points',
                                    fontsize=8, color=color)
        
        # Draw explorers
        for explorer in self.explorers:
            pos_2d = explorer['position'][:2].cpu().numpy()
            x = (pos_2d[0] + 5) * 10
            y = (pos_2d[1] + 5) * 10
            
            interest = explorer['interest_level']
            color = plt.cm.plasma(interest)
            
            self.ax_semantic.scatter(x, y, c=[color], s=30, marker='*', 
                                   edgecolors='white', linewidth=1)
            
            # Draw trail
            if len(explorer['trail']) > 1:
                trail_points = torch.stack(list(explorer['trail']))
                trail_x = (trail_points[:, 0].cpu().numpy() + 5) * 10
                trail_y = (trail_points[:, 1].cpu().numpy() + 5) * 10
                self.ax_semantic.plot(trail_x, trail_y, color=color, alpha=0.6, linewidth=1)
        
        self.ax_semantic.set_xlim(0, 100)
        self.ax_semantic.set_ylim(0, 100)
        
        # 2. Attention field
        self.ax_attention.set_title('🎯 Attention Field')
        combined_attention = torch.zeros(100, 100, device=self.device)
        
        for attractor in self.attractors:
            combined_attention += attractor.attention_field
        
        self.ax_attention.imshow(combined_attention.cpu().numpy(), cmap='hot', origin='lower')
        
        # 3. Thinking dynamics
        self.ax_dynamics.set_title('⚡ Thinking Dynamics')
        if len(self.thought_history) > 1:
            steps = [t['step'] for t in list(self.thought_history)[-10:]]
            attention = [t['attention_strength'] for t in list(self.thought_history)[-10:]]
            
            self.ax_dynamics.plot(steps, attention, 'b-', linewidth=2, label='Attention')
            self.ax_dynamics.set_ylabel('Attention Strength')
            self.ax_dynamics.legend()
            self.ax_dynamics.grid(True, alpha=0.3)
        
        # 4. Token generation
        self.ax_tokens.set_title('📝 Recent Tokens')
        self.ax_tokens.axis('off')
        
        if len(self.thought_history) > 0:
            recent_tokens = [t['token'] for t in list(self.thought_history)[-5:]]
            tokens_text = ' '.join(recent_tokens)
            self.ax_tokens.text(0.1, 0.5, f"Recent tokens:\n{tokens_text}", 
                              transform=self.ax_tokens.transAxes, fontfamily='monospace',
                              verticalalignment='center', fontsize=12)
        
        # 5. Attractor states
        self.ax_states.set_title('🧠 Attractor States')
        self.ax_states.axis('off')
        
        states_text = "ATTRACTOR STATES:\n"
        for i, attractor in enumerate(self.attractors):
            states_text += f"{attractor.current_concept}: {attractor.decision_state}\n"
            states_text += f"  Focus: {attractor.focus_strength:.3f}\n"
            states_text += f"  Curiosity: {attractor.curiosity:.3f}\n"
        
        self.ax_states.text(0.1, 0.9, states_text, transform=self.ax_states.transAxes,
                          fontfamily='monospace', verticalalignment='top', fontsize=9)
        
        # 6. Thought flow
        self.ax_flow.set_title('💭 Thought Flow')
        if len(self.thought_history) > 3:
            # Show focus levels over time
            recent_history = list(self.thought_history)[-8:]
            steps = [h['step'] for h in recent_history]
            
            colors = ['red', 'blue', 'green', 'purple']
            for i, concept in enumerate(['curiosity', 'analysis', 'creativity', 'focus']):
                focus_levels = [h['focus_levels'][i] for h in recent_history]
                color = colors[i]
                self.ax_flow.plot(steps, focus_levels, color=color, linewidth=2, 
                                label=concept, alpha=0.8)
            
            self.ax_flow.legend()
            self.ax_flow.set_ylabel('Focus Level')
            self.ax_flow.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def interactive_thinking_session(self):
        """Interactive session where you can watch the AI think"""
        
        print("\n🧠⚡ INTERACTIVE THINKING SESSION")
        print("=" * 50)
        print("Watch Phi think with semantic attractors!")
        print("Enter prompts to see live thinking dynamics.")
        print("Type 'quit' to exit.")
        print()
        
        while True:
            try:
                prompt = input("🤔 Your prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    continue
                
                # Generate with thinking
                response, thinking_states = self.generate_with_thinking(
                    prompt, 
                    max_length=15,
                    temperature=0.9
                )
                
                print(f"\n🎭 Final response: {response}")
                print(f"🧠 Thinking steps: {len(thinking_states)}")
                
                # Show final visualization
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n👋 Thinking session ended!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

def main():
    """Run the Phi + Attractor thinking system"""
    
    print("🧠⚡ PHI + ATTRACTOR THINKING SYSTEM")
    print("=" * 60)
    print("Initializing hybrid AI with semantic attractors...")
    print()
    
    try:
        # Initialize the hybrid system
        hybrid = PhiAttractorHybrid()
        
        print("\n🚀 System ready! Starting interactive thinking session...")
        print("Watch the visualization to see semantic attractors explore meaning space!")
        print()
        
        # Start interactive session
        hybrid.interactive_thinking_session()
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()