
class BudgetCalculator:

    def calculate_budget(
        self,
        cost_per_request,
        daily_users=1000,
        requests_per_user=1
    ):

        total_requests = daily_users * requests_per_user

        daily_cost = total_requests * cost_per_request
        monthly_cost = daily_cost * 30

        return {
            "daily_users": daily_users,
            "total_requests": total_requests,
            "daily_cost": daily_cost,
            "monthly_cost": monthly_cost
        }
