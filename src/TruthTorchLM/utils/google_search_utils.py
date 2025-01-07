import asyncio
import aiohttp
import os
import nest_asyncio
import ast



def extract_dict_from_string(input_string):
    start_index = input_string.find('{')
    end_index = input_string.rfind('}')

    if start_index != -1 and end_index != -1 and start_index < end_index:
        return input_string[start_index:end_index + 1]
    else:
        return None

def extract_list_from_string(input_string):
        start_index = input_string.find('[')  
        end_index = input_string.rfind(']') 

        if start_index != -1 and end_index != -1 and start_index < end_index:
            return input_string[start_index:end_index + 1]
        else:
            return None


def type_check(output, expected_type):
    try:
        output_eval = ast.literal_eval(output)
        if not isinstance(output_eval, expected_type):
            return None
        return output_eval
    except:
        return None



class GoogleSerperAPIWrapper():
    """Wrapper around the Serper.dev Google Search API.
    You can create a free API key at https://serper.dev.
    To use, you should have the environment variable ``SERPER_API_KEY``
    set with your API key, or pass `serper_api_key` as a named parameter
    to the constructor.
    Example:
        .. code-block:: python
            from langchain import GoogleSerperAPIWrapper
            google_serper = GoogleSerperAPIWrapper()
    """
    def __init__(self, snippet_cnt = 10, location:str = 'us', language = 'en') -> None:
        self.k = snippet_cnt
        self.gl = location
        self.hl = language
        self.serper_api_key = os.environ.get("SERPER_API_KEY", None)
        assert self.serper_api_key is not None, "Please set the SERPER_API_KEY environment variable."
        assert self.serper_api_key != '', "Please set the SERPER_API_KEY environment variable."
        nest_asyncio.apply()# to allow asyncio.run in jupyter notebook

    async def _google_serper_search_results(self, session, search_term: str, gl: str, hl: str) -> dict:
        headers = {
            "X-API-KEY": self.serper_api_key or "",
            "Content-Type": "application/json",
        }
        params = {"q": search_term, "gl": gl, "hl": hl}
        async with session.post(
            "https://google.serper.dev/search", headers=headers, params=params, raise_for_status=True
        ) as response:
            return await response.json()
    
    def _parse_results(self, results):
        snippets = []
        try:
            if results.get("answerBox"):
                answer_box = results.get("answerBox", {})
                if answer_box.get("answer"):
                    element = {"content":answer_box.get("answer"),"source":"None"}
                    return [element]
                elif answer_box.get("snippet"):
                    element = {"content":answer_box.get("snippet").replace("\n", " "),"source":"None"}
                    return [element]
                elif answer_box.get("snippetHighlighted"):
                    element = {"content":answer_box.get("snippetHighlighted"),"source":"None"}
                    return [element]
                
            if results.get("knowledgeGraph"):
                kg = results.get("knowledgeGraph", {})
                title = kg.get("title")
                entity_type = kg.get("type")
                if entity_type:
                    element = {"content":f"{title}: {entity_type}","source":"None"}
                    snippets.append(element)
                description = kg.get("description")
                if description:
                    element = {"content":description,"source":"None"}
                    snippets.append(element)
                for attribute, value in kg.get("attributes", {}).items():
                    element = {"content":f"{attribute}: {value}","source":"None"}
                    snippets.append(element)

            for result in results["organic"][: self.k]:
                if "snippet" in result:
                    element = {"content":result["snippet"],"source":result["link"]}
                    snippets.append(element)
                for attribute, value in result.get("attributes", {}).items():
                    element = {"content":f"{attribute}: {value}","source":result["link"]}
                    snippets.append(element)

            if len(snippets) == 0:
                element = {"content":"No good Google Search Result was found","source":"None"}
                return [element]
        except:
            element = {"content":"No good Google Search Result was found","source":"None"}
            return [element]
        
        # keep only the first k snippets
        snippets = snippets[:int(self.k / 2)]

        return snippets
    
    async def parallel_searches(self, search_queries, gl, hl):
        async with aiohttp.ClientSession() as session:
            tasks = [self._google_serper_search_results(session, query, gl, hl) for query in search_queries]
            search_results = await asyncio.gather(*tasks, return_exceptions=True)
            return search_results

    def run(self, queries:list[str]) -> list[list]:
        """Run query through GoogleSearch and parse result."""
        results = asyncio.run(self.parallel_searches(queries, gl=self.gl, hl=self.hl))
        snippets_list = []
        for i in range(len(results)):
            snippets_list.append(self._parse_results(results[i]))
        return snippets_list