from services.llm_service import get_llm_service
from utils.math_utils import cosine_similarity
from services.embedding_service import get_embedding_service
from repositories.element_repository import ElementRepository


class InstructionGeneratorService:
    def __init__(self, element_repository: ElementRepository):
        self.element_repository = element_repository
        self.embedding_service = get_embedding_service()
        self.llm_service = get_llm_service()

    async def retrieve_parts_simple(self, concept: str):
        return await self.element_repository.get_by_concept_simple(concept)
    
    async def retrieve_parts_semantic(self, concept: str):
        query_vec = self.embedding_service.embed(concept)
        elements = await self.element_repository.get_all()

        scored = []

        for e in elements:
            if not e.embedding:
                continue

            score = cosine_similarity(query_vec, e.embedding)
            scored.append((score, e))

        scored.sort(reverse=True, key=lambda x: x[0])
        print("Scored elements:", [(score, e.name) for score, e in scored[:10]])
        return [e for score, e in scored[:10] if score > 0.1]
    
    async def retrieve_parts(self, concept: str):
        exact_elements = await self.retrieve_parts_simple(concept)
        semantic_elements = await self.retrieve_parts_semantic(concept)

        scored = {}

        # exact search gets strong boost
        for element in exact_elements:
            scored[element.id] = {
                "element": element,
                "score": 1.0
            }

        # semantic search adds similarity
        query_vec = self.embedding_service.embed(concept)

        for element in semantic_elements:
            similarity = cosine_similarity(query_vec, element.embedding)

            if element.id not in scored:
                scored[element.id] = {
                    "element": element,
                    "score": 0
                }

            scored[element.id]["score"] += similarity

        sorted_elements = sorted(
            scored.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [x["element"] for x in sorted_elements[:10] if x["score"] > 0.1]
    
    async def generate_instruction(self, concept: str):
        parts = await self.retrieve_parts(concept)

        prompt = self.build_prompt(concept, parts)

        print(prompt)

        return self.llm_service.generate_instruction(prompt)
    
    def build_prompt(self, concept, parts):
        parts_text = "\n".join([
            f"""
            ID: {p.id}
            Name: {p.name}
            Description: {p.comment}
            """
            for p in parts
        ])

        return f"""
            You are an expert electronics engineer.

            Generate a step-by-step electronics project instruction.

            PROJECT:
            {concept}

            AVAILABLE PARTS:
            {parts_text}

            IMPORTANT RULES:
            - Prefer using available components
            - You may add additional components if required
            - Each step must contain:
                - step_number
                - description
                - part_ids
            - part_ids must reference AVAILABLE COMPONENT IDs
            - Return ONLY valid JSON
            - No markdown
            - No explanations
            - No text outside JSON

            JSON SCHEMA:

            {{
            "name": "Project name",
            "steps": [
                {{
                "step_number": 1,
                "description": "Step description",
                "part_ids": [1, 2]
                }}
            ]
            }}
        """