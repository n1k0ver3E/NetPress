import os
import subprocess

# Define the folder for storing policies

def deploy_policies(policy_names=None, root_dir=None):
    """Deploy policies to the Kubernetes cluster."""
    [ "network-policy-adservice", "network-policy-cartservice", "network-policy-checkoutservice", "network-policy-currencyservice", "network-policy-emailservice", "network-policy-frontend", "network-policy-loadgenerator", "network-policy-paymentservice", "network-policy-productcatalogservice", "network-policy-recommendationservice", "network-policy-redis", "network-policy-shippingservice", "network-policy-deny-all"]
    print(f"Deploying policies: {policy_names}")
    for name in policy_names:
        filename = os.path.join(root_dir, "policies", f"{name}.yaml")
        try:
            result = subprocess.run(["kubectl", "apply", "-f", filename], check=True, text=True, capture_output=True)
            print(f"Deployed {filename}:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to deploy {filename}:\n{e.stderr}")
