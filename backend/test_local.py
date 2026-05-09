"""
本地测试脚本 - 直接调用 EmoCare Agent
"""
import asyncio
from src.graph.agent import _get_agent


async def test_chat():
    """测试对话功能"""
    print("=" * 50)
    print("EmoCare Agent 本地测试")
    print("=" * 50)
    
    # 测试用例
    test_messages = [
        "今天心情不太好，工作压力很大",
        "我感觉很焦虑，睡不着觉",
        "谢谢你的陪伴，我好多了",
    ]
    
    for msg in test_messages:
        print(f"\n👤 用户: {msg}")
        
        result = await _get_agent().chat(
            user_input=msg,
            user_id="test_user",
            session_id="test_session"
        )
        
        print(f"🤖 小暖: {result['response']}")
        print(f"   情绪: {result.get('emotion', {}).get('emotion', 'N/A')} "
              f"(强度: {result.get('emotion', {}).get('intensity', 'N/A')})")
        print(f"   场景: {result.get('scene', 'N/A')}")
        print("-" * 50)


async def interactive_chat():
    """交互式对话"""
    print("=" * 50)
    print("EmoCare Agent 交互模式")
    print("输入 'quit' 退出")
    print("=" * 50)
    
    session_id = "interactive_session"
    
    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见！照顾好自己 💙")
                break
            
            if not user_input:
                continue
            
            result = await _get_agent().chat(
                user_input=user_input,
                user_id="local_user",
                session_id=session_id
            )
            
            print(f"\n🤖 小暖: {result['response']}")
            
            # 显示情绪信息
            emotion = result.get('emotion', {})
            if emotion:
                print(f"   [情绪: {emotion.get('emotion', '?')} | "
                      f"强度: {emotion.get('intensity', 0):.1%}]")
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # 交互模式
        asyncio.run(interactive_chat())
    else:
        # 测试模式
        asyncio.run(test_chat())
