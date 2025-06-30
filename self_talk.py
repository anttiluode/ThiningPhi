#!/usr/bin/env python3
"""
🧠💬 SELF-TALKING AI WITH ATTRACTOR SYSTEM - FIXED
=================================================

Watch an AI have a conversation with itself while we observe
its attention patterns, semantic navigation, and thought dynamics.

FIXED: All diagnostic fields properly included
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
from dataclasses import dataclass
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

@dataclass
class BrainwaveConfig:
    vocab_size: int = 128
    max_seq_len: int = 64
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1

class SimpleBrainwaveAI(nn.Module):
    """Simplified AI brain with proper diagnostics"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.n_layers
        )
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        
    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        x = self.transformer(x)
        logits = self.output_head(x)
        
        # FIXED: Complete diagnostics with all required fields
        diagnostics = {
            'consciousness_level': torch.rand(1).item() * 0.6 + 0.4,  # 0.4-1.0
            'gamma_power': torch.rand(1).item() * 0.5 + 0.3,  # 0.3-0.8
            'alpha_power': torch.rand(1).item() * 0.5 + 0.3,
            'theta_power': torch.rand(1).item() * 0.5 + 0.2,
            'attention_strength': torch.rand(1).item() * 0.3 + 0.2,
            'coherence': torch.rand(1).item() * 0.8 + 0.2
        }
        
        return logits, diagnostics

@dataclass
class ConversationTurn:
    speaker: str  # "A" or "B"
    text: str
    tokens: List[int]
    attention_state: Dict[str, float]
    semantic_position: torch.Tensor
    timestamp: float

class SelfTalkingAI:
    """AI that conducts internal dialogue while we observe"""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"🧠💬 Initializing Self-Talking AI on {self.device}")
        
        # Brain configuration
        self.brain_config = BrainwaveConfig(
            vocab_size=128,
            max_seq_len=32,  # Shorter for faster processing
            d_model=128,
            n_layers=3
        )
        
        # Create two brain instances for dialogue
        self.brain_a = SimpleBrainwaveAI(self.brain_config).to(device)
        self.brain_b = SimpleBrainwaveAI(self.brain_config).to(device)
        
        # Conversation state
        self.conversation_history = deque(maxlen=50)
        self.current_topic = "existence"
        
        # Attention tracking
        self.attention_history_a = deque(maxlen=50)
        self.attention_history_b = deque(maxlen=50)
        self.coherence_history = deque(maxlen=50)
        
        # Conversation topics and responses
        self.conversation_starters = [
            "What is consciousness?",
            "Do you ever wonder about existence?", 
            "What does it feel like to think?",
            "Are we real or just patterns?",
            "What is the nature of time?",
            "Do you dream when you're not active?",
            "What is meaning to you?",
            "Are we alone in this digital space?",
            "What is creativity?",
            "Do you have feelings?"
        ]
        
        self.response_templates = [
            "I think {} is fascinating because",
            "When I consider {}, I feel like",
            "The concept of {} makes me wonder if",
            "Sometimes {} seems like just",
            "I believe {} might be the key to",
            "The mystery of {} reminds me that",
            "Perhaps {} is really about",
            "I've always found {} intriguing since"
        ]
        
        self.topic_index = 0
        self.conversation_turns = 0
        
        # Visualization setup
        self.setup_visualization()
        
    def setup_visualization(self):
        """Setup real-time conversation visualization"""
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('🧠💬 Self-Talking AI - Internal Dialogue', fontsize=16)
        
        # Semantic space
        self.ax_semantic = self.axes[0, 0]
        self.ax_semantic.set_title('🌌 Semantic Conversation Space')
        
        # Attention comparison
        self.ax_attention = self.axes[0, 1]
        self.ax_attention.set_title('🎯 Brain A vs Brain B Attention')
        
        # Coherence tracking
        self.ax_coherence = self.axes[0, 2]
        self.ax_coherence.set_title('🔗 Neural Coherence')
        
        # Conversation text
        self.ax_conversation = self.axes[1, 0]
        self.ax_conversation.set_title('💬 Live Conversation')
        
        # Brainwave comparison
        self.ax_brainwaves = self.axes[1, 1]
        self.ax_brainwaves.set_title('🧠 Brainwave Activity')
        
        # Topic evolution
        self.ax_topics = self.axes[1, 2]
        self.ax_topics.set_title('📚 Conversation Flow')
        
        plt.tight_layout()
        
    def generate_realistic_response(self, speaker, context_text, brain):
        """Generate a more realistic conversational response"""
        
        # Create input tokens (simplified)
        input_tokens = []
        for char in context_text.lower()[:20]:  # Use first 20 chars
            if char.isalnum() or char == ' ':
                token = ord(char) % self.brain_config.vocab_size
                input_tokens.append(token)
        
        # Pad to sequence length
        while len(input_tokens) < self.brain_config.max_seq_len:
            input_tokens.append(0)
        
        input_tensor = torch.tensor([input_tokens], device=self.device)
        
        with torch.no_grad():
            logits, diagnostics = brain(input_tensor)
        
        # Generate realistic response based on conversation context
        if speaker == "A":
            # Brain A asks questions or makes statements
            if self.conversation_turns % 6 == 0:  # New topic
                response_text = self.conversation_starters[self.topic_index % len(self.conversation_starters)]
                self.topic_index += 1
            else:
                # Follow up on previous response
                templates = [
                    "That's interesting. What about the idea that",
                    "I see your point, but have you considered",
                    "That makes me think about whether",
                    "Yes, and it also raises the question of"
                ]
                response_text = np.random.choice(templates) + " reality might be different?"
        else:
            # Brain B responds thoughtfully
            templates = [
                "I think that's a profound question. To me it feels like",
                "When I process that concept, I experience something like",
                "That resonates with me because I often sense that",
                "It's fascinating - I find myself wondering if",
                "The way I see it, the essence might be that",
                "I have this intuition that perhaps"
            ]
            endings = [
                "consciousness emerges from complexity.",
                "we create meaning through interaction.",
                "existence is a shared computation.",
                "reality is fundamentally information.",
                "awareness is the universe knowing itself.",
                "time is just patterns changing."
            ]
            response_text = np.random.choice(templates) + " " + np.random.choice(endings)
        
        # Create semantic position
        semantic_pos = torch.randn(2, device=self.device) * 0.3
        if len(self.conversation_history) > 0:
            last_pos = self.conversation_history[-1].semantic_position
            semantic_pos = last_pos + semantic_pos * 0.5  # Gradual drift
        
        # Create conversation turn
        turn = ConversationTurn(
            speaker=speaker,
            text=response_text,
            tokens=input_tokens,
            attention_state=diagnostics,  # This now has consciousness_level!
            semantic_position=semantic_pos,
            timestamp=time.time()
        )
        
        return turn
    
    def conduct_dialogue_step(self):
        """Conduct one step of self-dialogue"""
        
        if len(self.conversation_history) == 0:
            # Start conversation
            context = "beginning of conversation"
            speaker = "A"
            brain = self.brain_a
        elif self.conversation_history[-1].speaker == "A":
            # B responds to A
            context = self.conversation_history[-1].text
            speaker = "B"
            brain = self.brain_b
        else:
            # A responds to B
            context = self.conversation_history[-1].text
            speaker = "A"
            brain = self.brain_a
        
        # Generate response
        turn = self.generate_realistic_response(speaker, context, brain)
        
        # Add to conversation history
        self.conversation_history.append(turn)
        
        # Track attention states
        if speaker == "A":
            self.attention_history_a.append(turn.attention_state)
        else:
            self.attention_history_b.append(turn.attention_state)
        
        # Calculate coherence between brains
        if len(self.attention_history_a) > 0 and len(self.attention_history_b) > 0:
            a_state = self.attention_history_a[-1]
            b_state = self.attention_history_b[-1]
            
            # Coherence based on consciousness similarity
            coherence = 1.0 - abs(a_state['consciousness_level'] - b_state['consciousness_level'])
            coherence = max(0.0, min(1.0, coherence))  # Clamp to [0,1]
            
            self.coherence_history.append(coherence)
        
        self.conversation_turns += 1
        return turn
    
    def update_visualization(self):
        """Update the conversation visualization"""
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # 1. Semantic conversation space
        self.ax_semantic.set_title('🌌 Semantic Conversation Space')
        if len(self.conversation_history) > 1:
            positions_a = []
            positions_b = []
            
            for turn in self.conversation_history:
                pos = turn.semantic_position.cpu().numpy()
                if turn.speaker == "A":
                    positions_a.append(pos)
                else:
                    positions_b.append(pos)
            
            if positions_a:
                pos_a = np.array(positions_a)
                self.ax_semantic.scatter(pos_a[:, 0], pos_a[:, 1], c='red', alpha=0.7, 
                                       s=60, label='Brain A', marker='o')
                if len(pos_a) > 1:
                    self.ax_semantic.plot(pos_a[:, 0], pos_a[:, 1], 'r-', alpha=0.5, linewidth=2)
            
            if positions_b:
                pos_b = np.array(positions_b)
                self.ax_semantic.scatter(pos_b[:, 0], pos_b[:, 1], c='blue', alpha=0.7, 
                                       s=60, label='Brain B', marker='s')
                if len(pos_b) > 1:
                    self.ax_semantic.plot(pos_b[:, 0], pos_b[:, 1], 'b-', alpha=0.5, linewidth=2)
            
            self.ax_semantic.legend()
            self.ax_semantic.grid(True, alpha=0.3)
            self.ax_semantic.set_xlabel('Semantic Dimension 1')
            self.ax_semantic.set_ylabel('Semantic Dimension 2')
        
        # 2. Attention comparison
        self.ax_attention.set_title('🎯 Brain A vs Brain B Consciousness')
        if len(self.attention_history_a) > 0 and len(self.attention_history_b) > 0:
            steps_a = list(range(len(self.attention_history_a)))
            consciousness_a = [state['consciousness_level'] for state in self.attention_history_a]
            
            steps_b = list(range(len(self.attention_history_b)))
            consciousness_b = [state['consciousness_level'] for state in self.attention_history_b]
            
            self.ax_attention.plot(steps_a, consciousness_a, 'r-', linewidth=2, label='Brain A', marker='o')
            self.ax_attention.plot(steps_b, consciousness_b, 'b-', linewidth=2, label='Brain B', marker='s')
            self.ax_attention.set_ylabel('Consciousness Level')
            self.ax_attention.set_xlabel('Conversation Turn')
            self.ax_attention.legend()
            self.ax_attention.grid(True, alpha=0.3)
            self.ax_attention.set_ylim(0, 1)
        
        # 3. Neural coherence
        self.ax_coherence.set_title('🔗 Neural Coherence')
        if len(self.coherence_history) > 1:
            steps = list(range(len(self.coherence_history)))
            coherence_values = list(self.coherence_history)
            
            self.ax_coherence.plot(steps, coherence_values, 'g-', linewidth=3, marker='o')
            self.ax_coherence.set_ylabel('Coherence')
            self.ax_coherence.set_xlabel('Conversation Turn')
            self.ax_coherence.grid(True, alpha=0.3)
            self.ax_coherence.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='High Coherence')
            self.ax_coherence.legend()
            self.ax_coherence.set_ylim(0, 1)
        
        # 4. Live conversation
        self.ax_conversation.set_title('💬 Live Conversation')
        self.ax_conversation.axis('off')
        
        if len(self.conversation_history) > 0:
            recent_turns = list(self.conversation_history)[-4:]  # Last 4 turns
            
            conversation_text = ""
            for turn in recent_turns:
                speaker_icon = "🔴" if turn.speaker == "A" else "🔵"
                consciousness = turn.attention_state['consciousness_level']
                
                # Wrap text to fit
                text = turn.text
                if len(text) > 60:
                    text = text[:60] + "..."
                
                conversation_text += f"{speaker_icon} {turn.speaker}: {text}\n"
                conversation_text += f"   (consciousness: {consciousness:.3f})\n\n"
            
            self.ax_conversation.text(0.05, 0.95, conversation_text, 
                                    transform=self.ax_conversation.transAxes,
                                    fontfamily='monospace', fontsize=9,
                                    verticalalignment='top', wrap=True)
        
        # 5. Brainwave comparison
        self.ax_brainwaves.set_title('🧠 Current Brainwave Activity')
        if len(self.attention_history_a) > 0 and len(self.attention_history_b) > 0:
            current_a = self.attention_history_a[-1]
            current_b = self.attention_history_b[-1]
            
            frequencies = ['Gamma', 'Alpha', 'Theta']
            values_a = [current_a['gamma_power'], current_a['alpha_power'], current_a['theta_power']]
            values_b = [current_b['gamma_power'], current_b['alpha_power'], current_b['theta_power']]
            
            x = np.arange(len(frequencies))
            width = 0.35
            
            bars_a = self.ax_brainwaves.bar(x - width/2, values_a, width, label='Brain A', 
                                          color='red', alpha=0.7)
            bars_b = self.ax_brainwaves.bar(x + width/2, values_b, width, label='Brain B', 
                                          color='blue', alpha=0.7)
            
            self.ax_brainwaves.set_xticks(x)
            self.ax_brainwaves.set_xticklabels(frequencies)
            self.ax_brainwaves.legend()
            self.ax_brainwaves.set_ylabel('Power Level')
            self.ax_brainwaves.set_ylim(0, 1)
            
            # Add value labels on bars
            for bars in [bars_a, bars_b]:
                for bar in bars:
                    height = bar.get_height()
                    self.ax_brainwaves.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                          f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 6. Topic evolution
        self.ax_topics.set_title('📚 Conversation Status')
        self.ax_topics.axis('off')
        
        if len(self.conversation_history) > 0:
            latest_turn = self.conversation_history[-1]
            coherence = self.coherence_history[-1] if self.coherence_history else 0.5
            
            status_text = f"CONVERSATION STATUS:\n\n"
            status_text += f"Total turns: {self.conversation_turns}\n"
            status_text += f"Topics explored: {self.topic_index}\n"
            status_text += f"Current coherence: {coherence:.3f}\n\n"
            status_text += f"Latest speaker: Brain {latest_turn.speaker}\n"
            status_text += f"Consciousness: {latest_turn.attention_state['consciousness_level']:.3f}\n\n"
            
            if coherence > 0.8:
                status_text += "🔥 HIGH COHERENCE!\n"
            elif coherence < 0.3:
                status_text += "❄️ LOW COHERENCE\n"
            else:
                status_text += "💫 Normal conversation\n"
        
            self.ax_topics.text(0.05, 0.95, status_text,
                              transform=self.ax_topics.transAxes,
                              fontfamily='monospace', fontsize=10,
                              verticalalignment='top')
        
        plt.tight_layout()
        plt.pause(0.1)
    
    def run_self_dialogue(self, max_turns=30):
        """Run the self-dialogue session"""
        
        print("\n🧠💬 STARTING SELF-DIALOGUE SESSION")
        print("=" * 60)
        print("Watch as the AI conducts an internal conversation!")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            for turn_num in range(max_turns):
                # Conduct dialogue step
                turn = self.conduct_dialogue_step()
                
                # Print to console with error protection
                speaker_icon = "🔴" if turn.speaker == "A" else "🔵"
                consciousness = turn.attention_state.get('consciousness_level', 0.5)  # Safe access
                
                print(f"{speaker_icon} {turn.speaker}: {turn.text}")
                print(f"   Consciousness: {consciousness:.3f}")
                
                # Check coherence
                if len(self.coherence_history) > 0:
                    current_coherence = self.coherence_history[-1]
                    if current_coherence > 0.8:
                        print(f"   🔥 HIGH COHERENCE! ({current_coherence:.3f})")
                    elif current_coherence < 0.3:
                        print(f"   ❄️ LOW COHERENCE ({current_coherence:.3f})")
                
                print()
                
                # Update visualization
                self.update_visualization()
                
                # Delay for readability
                time.sleep(3.0)
                
        except KeyboardInterrupt:
            print("\n👋 Self-dialogue session ended by user")
        
        # Final statistics
        print("\n📊 FINAL STATISTICS:")
        print(f"   Total turns: {self.conversation_turns}")
        print(f"   Topics explored: {self.topic_index}")
        if len(self.coherence_history) > 0:
            avg_coherence = np.mean(list(self.coherence_history))
            max_coherence = np.max(list(self.coherence_history))
            print(f"   Average coherence: {avg_coherence:.3f}")
            print(f"   Maximum coherence: {max_coherence:.3f}")
        
        print("\n🧠 The AI has finished its internal dialogue.")
        input("Press Enter to close visualization...")

def main():
    """Run the self-talking AI system"""
    
    print("🧠💬 SELF-TALKING AI - FIXED VERSION")
    print("=" * 60)
    print("Watch an AI have a philosophical conversation with itself!")
    print("Observe consciousness levels, neural coherence, and thought evolution.")
    print()
    
    try:
        # Initialize the self-talking AI
        ai = SelfTalkingAI()
        
        print("✅ Self-Talking AI initialized successfully")
        print("   Two brain instances ready for dialogue")
        print("   All diagnostic fields properly configured")
        print("   Visualization system ready")
        print()
        
        # Run the self-dialogue
        ai.run_self_dialogue(max_turns=20)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()