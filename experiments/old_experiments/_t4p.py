from data import get_benchmark_program
import tests4py.api as t4p
import os


subject = t4p.fastapi_1
report = t4p.checkout(subject)
if report.raised:
    print("ERROR")

print(report)

systemtests_path = os.path.join("tmp", subject.get_identifier(), "tests4py_systemtest_diversity")

failing = [
    os.path.abspath(os.path.join(systemtests_path, f"failing_test_diversity_{i}")) 
    for i in range(10)
]

passing = [
    os.path.abspath(os.path.join(systemtests_path, f"passing_test_diversity_{i}"))
    for i in range(10)
]

af = [
    os.path.abspath(os.path.join("repair_tests", "COOKIECUTTER_3_10-10_959", f"failing_test_{i}")) 
    for i in range(10)
]

ap = [
    os.path.abspath(os.path.join("repair_tests", "COOKIECUTTER_3_10-10_959", f"passing_test_{i}"))
    for i in range(10)
]

t4p.build()

#path = os.path.abspath("test.txt")

path = failing + passing
print(path)


output = "output_t4p.txt"

t4p.systemtest_test(report.location, diversity=True)
