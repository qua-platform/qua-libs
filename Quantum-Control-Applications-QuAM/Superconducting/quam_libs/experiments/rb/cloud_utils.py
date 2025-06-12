def write_sync_hook(circuits_as_ints_batched: list):

    with open("sync_hook.py", "w") as f:
        f.write("from iqcc_cloud_client.runtime import get_qm_job\n")
        f.write("from qm.qua import *\n\n")
        f.write("job = get_qm_job()\n")
        f.write("result = job.result_handles\n\n")
        f.write("# The actual values from circuits_as_ints_batched\n")
        f.write("circuits_as_ints_batched = [\n")
        for batch in circuits_as_ints_batched:
            f.write(f"    {batch},\n")
        f.write("]\n\n")
        f.write("for id, batch in enumerate(circuits_as_ints_batched):\n")
        f.write('    job.push_to_input_stream("sequence", batch)\n')
        f.write("    result.measurements.wait_for_values(id+1)\n")
        f.write('    print(f"{id}: Received ", str(result.measurements.fetch(id)))')