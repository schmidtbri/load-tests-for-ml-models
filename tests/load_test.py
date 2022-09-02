import logging
from locust import HttpUser, constant_throughput, task, events, LoadTestShape
from faker import Faker

logger = logging.getLogger(__name__)


class ModelServiceUser(HttpUser):
    wait_time = constant_throughput(1)

    @task
    def post_prediction(self):
        faker = Faker()

        record = {
            "age": faker.random_int(min=18, max=65),
            "sex": faker.random_choices(elements=("male", "female"), length=1)[0],
            "bmi": faker.random_int(min=15000, max=50000) / 1000.0,
            "children": faker.random_int(min=0, max=5),
            "smoker": faker.boolean(),
            "region": faker.random_choices(
                elements=("southwest", "southeast", "northwest", "northeast"), length=1)[0]
        }
        self.client.post("/api/models/insurance_charges_model/prediction", json=record)


class StagesShape(LoadTestShape):
    """Simple load test shape class."""

    stages = [
        {"duration": 30, "users": 1, "spawn_rate": 1},
        {"duration": 60, "users": 2, "spawn_rate": 1},
        {"duration": 90, "users": 3, "spawn_rate": 1},
        {"duration": 120, "users": 4, "spawn_rate": 1},
        {"duration": 150, "users": 5, "spawn_rate": 1},
        {"duration": 180, "users": 4, "spawn_rate": 1},
        {"duration": 210, "users": 3, "spawn_rate": 1},
        {"duration": 240, "users": 2, "spawn_rate": 1},
        {"duration": 270, "users": 1, "spawn_rate": 1}
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                tick_data = (stage["users"], stage["spawn_rate"])
                return tick_data
        # returning None to stop the load test
        return None


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    process_exit_code = 0

    max_requests_per_second = max(
        [requests_per_second for requests_per_second in environment.stats.total.num_reqs_per_sec.values()])

    if environment.stats.total.fail_ratio > 0.0:
        logger.error("Test failed because there was one or more errors.")
        process_exit_code = 1

    if environment.stats.total.get_response_time_percentile(0.99) > 100:
        logger.error("Test failed because the response time at the 99th percentile was above 100 ms. The 99th "
                     "percentile latency is '{}'.".format(environment.stats.total.get_response_time_percentile(0.99)))
        process_exit_code = 1

    if max_requests_per_second < 5:
        logger.error(
            "Test failed because the max requests per second never reached 5. The max requests per second "
            "is: '{}'.".format(max_requests_per_second))
        process_exit_code = 1

    environment.process_exit_code = process_exit_code
