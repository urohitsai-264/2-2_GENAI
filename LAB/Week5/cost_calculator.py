
class CostCalculator:

    # Pricing per 1 million tokens (USD)
    PRICING = {
        "gemini-3.5-flash-lite": {
            "input": 0.30,
            "output": 2.50
        }
    }

    def calculate_cost(self, model, input_tokens, output_tokens):

        if model not in self.PRICING:
            return {
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0
            }

        price = self.PRICING[model]

        input_cost = (input_tokens / 1_000_000) * price["input"]
        output_cost = (output_tokens / 1_000_000) * price["output"]

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost
        }
