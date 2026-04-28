from typing import List, Dict, Deque,TypedDict
from collections import deque
# ==========================================
# 1. 严格复制参考文件中的模板常量
# ==========================================

active_thought_prompt_template = '''Your environment description contains the following points:[\n{}\n]
Your role description contains the following properties:[\n{}\n]
Your chat history contains the following records:[\n{}\n]
Your album contains the following images:[\n{}\n]'''

active_thought_prompt_q = "Behave as you are {}. let's say hello first and Please select an image from your album and explain why.You should choose the image that best fits your personality and the current environment. "

active_action_prompt_template = '''Your environment description contains the following points:[\n{}\n]
Your role description contains the following properties:[\n{}\n]
Your chat history contains the following records:[\n{}\n]'''

active_action_prompt_q = "<image>\nBehave as you are {}. Please ask a question about the image." 

passive_action_prompt_template = '''Your environment description contains the following points:[\n{}\n]
Your role description contains the following properties:[\n{}\n]
Your chat history contains the following records:[\n{}\n]'''

passive_action_prompt_q = "<image>\nBehave as you are {}. {}"

# LLaVA v1 系统默认提示词
LLAVA_SYSTEM_PROMPT = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

# ==========================================
# 2. 提示词生成工具类
# ==========================================

class PromptGenerator:
    @staticmethod
    def _format_llava_prompt(system_text: str, user_content: str) -> str:
        """
        模拟 conv_llava_v1 (SeparatorStyle.TWO) 的格式化逻辑
        格式: System + " " + "USER: " + message + " " + "ASSISTANT:"
        注意：参考代码中使用了 sep2="</s>" 但通常在 prompt 构造时只用到 sep=" " 和 Role 标记
        """
        # 参考代码逻辑：state.system = state.system + "\n" + template
        full_system = f"{LLAVA_SYSTEM_PROMPT}\n{system_text}"
        
        # 构造最终字符串，严格模仿 Conversation.get_prompt() 对于 LLaVA v1 的行为
        # 格式通常为: System <SEP> USER: <Content> <SEP> ASSISTANT:
        # 参考代码 sep=" "
        return f"{full_system} USER: {user_content} ASSISTANT:"

    @staticmethod
    def _process_album(album: Deque) -> List[str]:
        """
        严格复刻参考代码中的相册处理逻辑：
        album_list = [": ".join(i.split("/")[-1].split("_")[0:2]) for i in self.album]
        """
        processed = []
        for i in album:
            try:
                # 提取文件名 -> 分割下划线 -> 取前两部分 -> 冒号连接
                # 例如: path/to/cat_playing_001.jpg -> cat: playing
                name_part = i.split("/")[-1]
                parts = name_part.split("_")[0:2]
                processed.append(": ".join(parts))
            except:
                # 如果格式不符合预期，保留原名以防报错
                processed.append(str(i))
        return processed

    @staticmethod
    def get_prompts(agent_state: Dict, agent_name: str, env_description: List[str]):
        """
        生成 Active Thought, Active Action 和 Passive Action 提示词
        
        Args:
            agent_state: 你的 AgentInternalState 对象
            agent_name: 代理名称 (用于填充 {})
            env_description: 环境描述列表
        Returns:
            dict: 包含三个生成的 prompt
        """
        
        # 1. 准备数据
        # 获取当前人格 (假设使用 list 中的第一个或者是 main_persona_idx 指定的)
        current_persona = agent_state['personas'][0] # 或者使用 agent_state['personas'][agent_state['main_persona_idx']]
        
        # 将人格字典转换为字符串列表 (参考代码需要 list join)
        role_desc_list = [f"{k}: {v}" for k, v in current_persona.items()]
        
        # 获取聊天历史 (参考代码取最后 max_records 条)
        # 这里假设 max_records 逻辑在外部控制，或者取全部 deque
        chat_history_list = list(agent_state['chat_history'])
        
        # 处理相册
        album_processed = PromptGenerator._process_album(agent_state['photo_album'])

        # -------------------------------------------------
        # 2. 生成 Active Thought Prompt (思考/选图阶段)
        # -------------------------------------------------
        # 填充模板
        thought_sys_add = active_thought_prompt_template.format(
            "\n".join(env_description),
            "\n".join(role_desc_list),
            "\n".join(chat_history_list),
            "\n".join(album_processed)
        )
        thought_q = active_thought_prompt_q.format(agent_name)
        
        # 格式化
        active_thought_prompt = PromptGenerator._format_llava_prompt(thought_sys_add, thought_q)

        # -------------------------------------------------
        # 3. 生成 Active Action Prompt (主动提问阶段)
        # -------------------------------------------------
        action_sys_add = active_action_prompt_template.format(
            "\n".join(env_description),
            "\n".join(role_desc_list),
            "\n".join(chat_history_list)
        )
        action_q = active_action_prompt_q.format(agent_name)
        
        # 格式化
        active_action_prompt = PromptGenerator._format_llava_prompt(action_sys_add, action_q)

        return {
            "active_thought": active_thought_prompt,
            "active_action": active_action_prompt
        }

    @staticmethod
    def get_passive_response_prompt(agent_state: Dict, agent_name: str, env_description: List[str], incoming_question: str):
        """
        生成 Passive Action Prompt (被动回复阶段)
        """
        current_persona = agent_state['personas'][0]
        role_desc_list = [f"{k}: {v}" for k, v in current_persona.items()]
        chat_history_list = list(agent_state['chat_history'])

        # 填充模板
        passive_sys_add = passive_action_prompt_template.format(
            "\n".join(env_description),
            "\n".join(role_desc_list),
            "\n".join(chat_history_list)
        )
        
        # 填充问题
        passive_q = passive_action_prompt_q.format(agent_name, incoming_question)
        
        # 格式化
        passive_action_prompt = PromptGenerator._format_llava_prompt(passive_sys_add, passive_q)
        
        return passive_action_prompt
class AgentInternalState(TypedDict):
    chat_history: deque                    # 与其他代理的聊天记录
    photo_album: deque                     # 相册
    personas: List[Dict]                   # 人格列表   主人格索引为0
    self_interaction_history: List[str]    # 自交互历史 (用于 Prompt 上下文)
    
    # 用于辅助计算指标的临时日志 (CLIP分析用)
    
    metrics_log: Dict[str, List[str]]