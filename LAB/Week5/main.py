
import time

from llm_client import LLMClient
from cost_calculator import CostCalculator
from budget_calculator import BudgetCalculator
from rate_limit_checker import RateLimitChecker


def main():

    print("\n" + "=" * 45)
    print("       LLM TOKEN ECONOMICS ANALYZER")
    print("=" * 45)

    provider = input("\nChoose provider (Gemini/Ollama): ").strip().lower()

    if provider not in ["gemini", "ollama"]:
        print("Invalid provider. Please enter Gemini or Ollama.")
        return

    rate_limiter = RateLimitChecker()

    while True:

        prompt = input("\nEnter your prompt (or type exit): ").strip()

        if prompt.lower() == "exit":
            print("\nThank you for using LLM Token Economics Analyzer!")
            break

        if not prompt:
            print("Prompt cannot be empty.")
            continue

        estimated_tokens = max(1, len(prompt) // 4)

        allowed, message = rate_limiter.check_limit(estimated_tokens)

        if not allowed:
            print(f"\nRequest blocked: {message}")
            print("Please wait before trying again.")
            continue

        try:
            print(f"\nConnecting to {provider.capitalize()}...")
            print("Generating response...\n")

            llm = LLMClient(provider)

            start_time = time.time()
            result = llm.generate_response(prompt)
            latency = time.time() - start_time

            input_tokens = result["input_tokens"]
            output_tokens = result["output_tokens"]
            total_tokens = input_tokens + output_tokens

            rate_limiter.record_usage(total_tokens)

            cost_calculator = CostCalculator()

            if provider == "gemini":
                costs = cost_calculator.calculate_cost(
                    result["model"],
                    input_tokens,
                    output_tokens
                )
            else:
                costs = {
                    "input_cost": 0.0,
                    "output_cost": 0.0,
                    "total_cost": 0.0
                }

            budget_calculator = BudgetCalculator()

            budget = budget_calculator.calculate_budget(
                costs["total_cost"]
            )

            print("=" * 45)
            print("MODEL RESPONSE")
            print("=" * 45)
            print(result["response"])

            print("\n" + "=" * 45)
            print("TOKEN ANALYSIS")
            print("=" * 45)

            print(f"Provider          : {result['provider']}")
            print(f"Model             : {result['model']}")
            print(f"Input Tokens      : {input_tokens}")
            print(f"Output Tokens     : {output_tokens}")
            print(f"Total Tokens      : {total_tokens}")
            print(f"Response Latency  : {latency:.2f} seconds")

            print("\n" + "=" * 45)
            print("COST ANALYSIS")
            print("=" * 45)

            print(f"Input Cost        : ${costs['input_cost']:.8f}")
            print(f"Output Cost       : ${costs['output_cost']:.8f}")
            print(f"Total Cost        : ${costs['total_cost']:.8f}")

            print("\n" + "=" * 45)
            print("BUDGET FOR 1000 DAILY USERS")
            print("=" * 45)

            print(f"Daily Users       : {budget['daily_users']}")
            print(f"Daily Requests    : {budget['total_requests']}")
            print(f"Daily Cost        : ${budget['daily_cost']:.6f}")
            print(f"Monthly Cost      : ${budget['monthly_cost']:.6f}")

            print("\n" + "=" * 45)
            print("RATE LIMIT ANALYSIS")
            print("=" * 45)

            status = rate_limiter.get_status()

            print(f"Requests Used       : {status['requests_used']}")
            print(f"RPM Limit           : {status['rpm_limit']}")
            print(f"Remaining Requests  : {status['remaining_requests']}")

            print(f"Tokens Used         : {status['tokens_used']}")
            print(f"TPM Limit           : {status['tpm_limit']}")
            print(f"Remaining Tokens    : {status['remaining_tokens']}")

            print("Request Status      : Within Configured Limits")

            print("\n" + "=" * 45)
            print("Analysis completed successfully!")
            print("=" * 45)

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()
