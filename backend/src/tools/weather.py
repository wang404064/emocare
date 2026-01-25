"""
天气查询工具 (Weather Tool)
提供天气信息查询功能
"""
from typing import Dict, Any, Optional
from loguru import logger
import httpx
import os


class WeatherTool:
    """天气查询工具"""
    
    def __init__(self):
        self.name = "weather"
        self.description = "查询指定城市的天气信息"
        # 可以使用免费的天气API，如OpenWeatherMap、和风天气等
        # 这里使用示例API，实际使用时需要配置API密钥
        self.api_key = os.getenv("WEATHER_API_KEY", "")
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    async def get_weather(self, city: str, units: str = "metric") -> Dict[str, Any]:
        """
        获取城市天气信息
        
        Args:
            city: 城市名称（支持中文和英文）
            units: 单位制（metric=摄氏度, imperial=华氏度）
        
        Returns:
            天气信息字典
        """
        if not self.api_key:
            # 如果没有配置API密钥，返回模拟数据
            logger.warning("天气API密钥未配置，返回模拟数据")
            return self._get_mock_weather(city)
        
        try:
            # 使用OpenWeatherMap API（需要注册获取API key）
            async with httpx.AsyncClient(timeout=10.0) as client:
                params = {
                    "q": city,
                    "appid": self.api_key,
                    "units": units,
                    "lang": "zh_cn"
                }
                response = await client.get(self.base_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    return self._format_weather_data(data, city)
                else:
                    logger.error(f"天气API请求失败: {response.status_code}")
                    return self._get_mock_weather(city)
        
        except Exception as e:
            logger.error(f"获取天气信息失败: {e}")
            return self._get_mock_weather(city)
    
    def _format_weather_data(self, data: Dict, city: str) -> Dict[str, Any]:
        """格式化天气API返回的数据"""
        main = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        wind = data.get("wind", {})
        
        return {
            "city": city,
            "temperature": main.get("temp", 0),
            "feels_like": main.get("feels_like", 0),
            "humidity": main.get("humidity", 0),
            "pressure": main.get("pressure", 0),
            "description": weather.get("description", ""),
            "main": weather.get("main", ""),
            "wind_speed": wind.get("speed", 0),
            "wind_direction": wind.get("deg", 0),
            "visibility": data.get("visibility", 0) / 1000 if data.get("visibility") else None,
            "timestamp": data.get("dt", 0)
        }
    
    def _get_mock_weather(self, city: str) -> Dict[str, Any]:
        """返回模拟天气数据（用于测试或API不可用时）"""
        return {
            "city": city,
            "temperature": 22,
            "feels_like": 24,
            "humidity": 65,
            "pressure": 1013,
            "description": "晴朗",
            "main": "Clear",
            "wind_speed": 3.5,
            "wind_direction": 180,
            "visibility": 10,
            "timestamp": 0,
            "note": "这是模拟数据，请配置WEATHER_API_KEY以获取真实天气"
        }
    
    def format_weather_message(self, weather_data: Dict[str, Any]) -> str:
        """格式化天气信息为可读文本"""
        city = weather_data.get("city", "未知城市")
        temp = weather_data.get("temperature", 0)
        feels_like = weather_data.get("feels_like", 0)
        description = weather_data.get("description", "")
        humidity = weather_data.get("humidity", 0)
        wind_speed = weather_data.get("wind_speed", 0)
        
        lines = [
            f"🌤️ **{city}的天气**",
            "",
            f"温度: {temp}°C (体感 {feels_like}°C)",
            f"天气: {description}",
            f"湿度: {humidity}%",
            f"风速: {wind_speed} m/s"
        ]
        
        if weather_data.get("note"):
            lines.append(f"\n*{weather_data['note']}*")
        
        return "\n".join(lines)
    
    async def run(self, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行天气查询工具
        
        Parameters:
            city: 城市名称（必需）
            units: 单位制（可选，默认metric）
        """
        params = parameters or {}
        city = params.get("city", "").strip()
        
        # 如果用户没有指定城市，返回需要询问用户的结果
        if not city:
            logger.info("用户未指定城市，需要询问用户")
            return {
                "success": False,
                "tool_name": self.name,
                "error": "需要城市信息",
                "needs_user_input": True,
                "question": "请问你想查询哪个城市的天气呢？"
            }
        
        logger.info(f"查询天气: {city}")
        
        units = params.get("units", "metric")
        weather_data = await self.get_weather(city, units)
        formatted_message = self.format_weather_message(weather_data)
        
        return {
            "success": True,
            "tool_name": self.name,
            "result": {
                "weather_data": weather_data,
                "formatted_message": formatted_message
            }
        }


# 全局实例
weather_tool = WeatherTool()
