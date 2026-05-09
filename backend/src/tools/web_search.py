"""
互联网搜索工具 (Web Search Tool)
提供互联网搜索功能（DuckDuckGo 免费优先，Google/Bing 可选）
"""
from typing import Dict, Any, List, Optional
from loguru import logger
import httpx
import os
import re


class WebSearchTool:
    """互联网搜索工具"""

    def __init__(self):
        self.name = "web_search"
        self.description = "在互联网上搜索信息"
        self.search_api = os.getenv("SEARCH_API", "duckduckgo")
        self.google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY", "")
        self.google_cx = os.getenv("GOOGLE_SEARCH_CX", "")
        self.bing_api_key = os.getenv("BING_SEARCH_API_KEY", "")
        self.max_results = 5

    async def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        使用 DuckDuckGo 搜索（双路：Instant Answer + HTML 抓取回退）
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # 第1路：Instant Answer API（百科摘要）
                resp = await client.get("https://api.duckduckgo.com/", params={
                    "q": query, "format": "json", "no_html": "1", "skip_disambig": "1"
                })
                results = []
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("AbstractText"):
                        results.append({
                            "title": data.get("Heading", query),
                            "snippet": data.get("AbstractText", ""),
                            "url": data.get("AbstractURL", ""),
                            "source": "DuckDuckGo"
                        })
                    for topic in data.get("RelatedTopics", [])[:max_results - 1]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            text = topic.get("Text", "")
                            title = text.split(" - ")[0] if " - " in text else text
                            results.append({
                                "title": title,
                                "snippet": text,
                                "url": topic.get("FirstURL", ""),
                                "source": "DuckDuckGo"
                            })
                # 第2路：Lite HTML 搜索回退（真实网页结果）
                if len(results) < 2:
                    html_results = await self._search_duckduckgo_html(client, query, max_results)
                    # 去重合并
                    existing_urls = {r["url"] for r in results if r["url"]}
                    for hr in html_results:
                        if hr["url"] not in existing_urls:
                            results.append(hr)
                            existing_urls.add(hr["url"])
                return results[:max_results]
        except Exception as e:
            logger.error(f"DuckDuckGo 搜索失败: {e}")
        return []

    async def _search_duckduckgo_html(self, client: httpx.AsyncClient, query: str, max_results: int) -> List[Dict[str, Any]]:
        """从 DuckDuckGo Lite HTML 页面提取搜索结果"""
        try:
            resp = await client.get("https://lite.duckduckgo.com/lite/", params={"q": query})
            if resp.status_code != 200:
                return []
            html = resp.text
            results = []
            # 解析 lite 版本的结果行：<a rel="nofollow" href="...">title</a><span>snippet</span>
            links = re.findall(r'<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', html)
            snippets = re.findall(r'<span class="link-snippet[^"]*">([^<]+)</span>', html)
            for i, (url, title) in enumerate(links):
                if i >= max_results:
                    break
                snippet = snippets[i] if i < len(snippets) else ""
                if url.startswith("//"):
                    url = "https:" + url
                results.append({
                    "title": title.strip(),
                    "snippet": snippet.strip()[:300],
                    "url": url,
                    "source": "DuckDuckGo"
                })
            return results
        except Exception:
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
