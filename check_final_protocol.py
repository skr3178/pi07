"""Guard checks for final-start isolation and immutable inference settings."""
import json
from pathlib import Path
import tempfile
from freeze_protocol import digest,spec_digest,validate_reserved_spec,validate_frozen_protocol


def rejected(call):
    try:call()
    except (ValueError,FileNotFoundError):return
    raise AssertionError("Invalid final protocol was accepted")


def main():
    spec={"phase":"final","suite":"libero_10","task_ids":list(range(10)),
          "init_ids":list(range(2,12)),"conditions":[{"name":"A","goals":"off"}],"max_steps":520}
    validate_reserved_spec(spec)
    for ids in ([0],[1],[12],[2,2],[]):rejected(lambda:validate_reserved_spec({**spec,"init_ids":ids}))
    rejected(lambda:validate_reserved_spec({**spec,"record_training_data":True}))
    rejected(lambda:validate_reserved_spec({**spec,"conditions":[{"action_noise_std":.1}]}))
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);asset=root/"asset";asset.write_bytes(b"frozen input")
        protocol=root/"protocol.json"
        protocol.write_text(json.dumps({"format":"pi07_frozen_final_v1","spec_sha256":spec_digest(spec),
                                        "files_sha256":{str(asset):digest(asset)}}))
        validate_frozen_protocol(spec,protocol)
        rejected(lambda:validate_frozen_protocol({**spec,"max_steps":521},protocol))
        asset.write_bytes(b"changed input");rejected(lambda:validate_frozen_protocol(spec,protocol))
        asset.unlink();rejected(lambda:validate_frozen_protocol(spec,protocol))
    result={"status":"passed","reserved_starts":list(range(2,12)),"rejections_checked":10,
            "limits":"Fixture checks of guards; no final simulator episodes executed."}
    target=Path(__file__).resolve().parent/"artifacts/final_protocol_checks_20261104.json"
    target.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))


if __name__=="__main__":main()
