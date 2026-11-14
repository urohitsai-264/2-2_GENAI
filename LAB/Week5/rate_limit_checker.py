
import os
import time
from collections import deque
from dotenv import load_dotenv

load_dotenv()


class RateLimitChecker:

    def __init__(self):

        self.rpm_limit = int(os.getenv("RATE_LIMIT_RPM", 10))
        self.tpm_limit = int(os.getenv("RATE_LIMIT_TPM", 20000))

        self.request_history = deque()
        self.token_history = deque()

    def clean_old_records(self):

        current_time = time.time()
        cutoff = current_time - 60

        while self.request_history and self.request_history[0] <= cutoff:
            self.request_history.popleft()

        while self.token_history and self.token_history[0][0] <= cutoff:
            self.token_history.popleft()

    def check_limit(self, estimated_tokens=0):

        self.clean_old_records()

        requests_used = len(self.request_history)

        tokens_used = sum(
            tokens for _, tokens in self.token_history
        )

        if requests_used >= self.rpm_limit:
            return False, "Requests per minute limit exceeded"

        if tokens_used + estimated_tokens > self.tpm_limit:
            return False, "Tokens per minute limit exceeded"

        return True, "Within configured limits"

    def record_usage(self, total_tokens):

        current_time = time.time()

        self.request_history.append(current_time)
        self.token_history.append((current_time, total_tokens))

    def get_status(self):

        self.clean_old_records()

        requests_used = len(self.request_history)

        tokens_used = sum(
            tokens for _, tokens in self.token_history
        )

        return {
            "requests_used": requests_used,
            "tokens_used": tokens_used,
            "rpm_limit": self.rpm_limit,
            "tpm_limit": self.tpm_limit,
            "remaining_requests": max(0, self.rpm_limit - requests_used),
            "remaining_tokens": max(0, self.tpm_limit - tokens_used)
        }
