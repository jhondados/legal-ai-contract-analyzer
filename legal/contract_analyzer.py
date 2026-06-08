"""AI contract analyzer for Brazilian law."""
from langchain_google_vertexai import ChatVertexAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from typing import List, Dict

RISK_PROMPT = """Analyze this contract clause and return JSON:
{
  "clause_type": "liability|payment|termination|ip|lgpd|other",
  "risk_level": "low|medium|high|critical",
  "risk_score": 0-10,
  "summary": "one sentence",
  "issues": ["list of issues"],
  "suggestions": ["list of improvements"],
  "lgpd_compliance": true|false
}"""

class LegalAIAnalyzer:
    def __init__(self):
        self.llm = ChatVertexAI(model_name="gemini-1.5-pro-002", temperature=0)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    def analyze_contract(self, contract_text: str) -> Dict:
        clauses = self.splitter.split_text(contract_text)
        results = []
        for clause in clauses:
            response = self.llm.invoke(f"{RISK_PROMPT}\n\nClause:\n{clause}",
                config={"response_format": {"type": "json_object"}})
            try: results.append(json.loads(response.content))
            except: pass
        critical = [r for r in results if r.get("risk_level") == "critical"]
        overall_risk = sum(r.get("risk_score", 5) for r in results) / max(len(results), 1)
        return {"clauses": results, "total_clauses": len(results), "critical_issues": critical,
                "overall_risk_score": round(overall_risk, 1), "lgpd_compliant": all(r.get("lgpd_compliance") for r in results)}
