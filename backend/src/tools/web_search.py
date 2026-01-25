"""
互联网搜索工具 (Web Search Tool)
提供互联网搜索功能
"""
from typing import Dict, Any, List, Optional
from loguru import logger
import httpx
import os
from urllib.parse import quote


class WebSearchTool:
    """互联网搜索工具"""
    
    def __init__(self):
        self.name = "web_search"
        self.description = "在互联网上搜索信息"
        # 可以使用多种搜索API，如Google Custom Search、Bing Search、DuckDuckGo等
        # 这里提供DuckDuckGo（免费）和Google Custom Search（需要API key）两种方案
        self.search_api = os.getenv("SEARCH_API", "duckduckgo")  # duckduckgo 或 google
        self.google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY", "")
        self.google_cx = os.getenv("GOOGLE_SEARCH_CX", "")
        self.max_results = 5
    
    async def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        使用DuckDuckGo搜索（免费，无需API key）
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
        
        Returns:
            搜索结果列表
        """
        try:
            # 使用DuckDuckGo Instant Answer API
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = "https://api.duckduckgo.com/"
                params = {
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1"
                }
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    # 提取摘要
                    if data.get("AbstractText"):
                        results.append({
                            "title": data.get("Heading", query),
                            "snippet": data.get("AbstractText", ""),
                            "url": data.get("AbstractURL", ""),
                            "source": "DuckDuckGo"
                        })
                    
                    # 提取相关主题
                    for topic in data.get("RelatedTopics", [])[:max_results-1]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append({
                                "title": topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else topic.get("Text", ""),
                                "snippet": topic.get("Text", ""),
                                "url": topic.get("FirstURL", ""),
                                "source": "DuckDuckGo"
                            })
                    
                    return results[:max_results]
        
        except Exception as e:
            logger.error(f"DuckDuckGo搜索失败: {e}")
        
        return []
    
    async def search_google(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        使用Google Custom Search API
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
        
        Returns:
            搜索结果列表
        """
        if not self.google_api_key or not self.google_cx:
            logger.warning("Google搜索API密钥未配置")
            return []
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    "key": self.google_api_key,
                    "cx": self.google_cx,
                    "q": query,
                    "num": min(max_results, 10)
                }
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    for item in data.get("items", [])[:max_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "url": item.get("link", ""),
                            "source": "Google"
                        })
                    
                    return results
        
        except Exception as e:
            logger.error(f"Google搜索失败: {e}")
        
        return []
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        执行搜索
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
        
        Returns:
            搜索结果列表
        """
        if self.search_api == "google" and self.google_api_key:
            return await self.search_google(query, max_results)
        else:
            return await self.search_duckduckgo(query, max_results)
    
    def format_search_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """格式化搜索结果为可读文本"""
        if not results:
            return f"抱歉，没有找到关于「{query}」的相关信息。"
        
        lines = [
            f"🔍 **搜索结果：{query}**",
            "",
            f"找到 {len(results)} 条相关信息：",
            ""
        ]
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "无标题")
            snippet = result.get("snippet", "")
            url = result.get("url", "")
            
            lines.append(f"{i}. **{title}**")
            if snippet:
                # 截断过长的摘要
                snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet
                lines.append(f"   {snippet}")
            if url:
                lines.append(f"   🔗 {url}")
            lines.append("")
        
        return "\n".join(lines)
    
    async def run(self, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行搜索工具
        
        Parameters:
            query: 搜索关键词（必需）
            max_results: 最大返回结果数（可选，默认5）
        """
        params = parameters or {}
        query = params.get("query", "")
        
        if not query:
            return {
                "success": False,
                "tool_name": self.name,
                "error": "缺少必需参数: query"
            }
        
        logger.info(f"执行搜索: {query}")
        
        max_results = params.get("max_results", self.max_results)
        results = await self.search(query, max_results)
        formatted_message = self.format_search_results(results, query)
        
        return {
            "success": True,
            "tool_name": self.name,
            "result": {
                "query": query,
                "results": results,
                "formatted_message": formatted_message,
                "result_count": len(results)
            }
        }


# 全局实例
web_search_tool = WebSearchTool()
