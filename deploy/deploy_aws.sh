#!/usr/bin/env bash
# Build, push and deploy the API to AWS App Runner via ECR.
#
# Prereqs: AWS CLI v2 configured, Docker running, models/ trained
#          (python -m shr.train). On Windows run from Git Bash or WSL.
# Usage:   ./deploy/deploy_aws.sh
#          AWS_REGION=eu-west-1 APP_NAME=my-api ./deploy/deploy_aws.sh
#
# ponytail: App Runner is the smallest AWS unit that runs a container with
# TLS, autoscaling and health checks — no VPC/ALB/cluster boilerplate.
# Outgrowing it (VPC-only data sources, >4 vCPU, sidecars) means graduating
# to ECS Fargate behind an ALB.
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
APP_NAME="${APP_NAME:-shr-offender-api}"
ROLE_NAME="${ROLE_NAME:-AppRunnerECRAccessRole}"
CPU="${CPU:-1024}"       # 1 vCPU
MEMORY="${MEMORY:-2048}" # 2 GB

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE="${ECR}/${APP_NAME}:latest"

echo "==> ECR repository ${APP_NAME}"
aws ecr describe-repositories --repository-names "$APP_NAME" --region "$AWS_REGION" >/dev/null 2>&1 ||
  aws ecr create-repository --repository-name "$APP_NAME" --region "$AWS_REGION" >/dev/null

echo "==> Build and push ${IMAGE}"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR"
docker build -t "$IMAGE" .
docker push "$IMAGE"

echo "==> IAM role for App Runner -> ECR access"
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow",
                   "Principal": {"Service": "build.apprunner.amazonaws.com"},
                   "Action": "sts:AssumeRole"}]
  }' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
  echo "    created ${ROLE_NAME}, waiting for IAM propagation"
  sleep 15
fi
ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)

SERVICE_ARN=$(aws apprunner list-services --region "$AWS_REGION" \
  --query "ServiceSummaryList[?ServiceName=='${APP_NAME}'].ServiceArn" --output text)

if [ -z "$SERVICE_ARN" ]; then
  echo "==> Creating App Runner service ${APP_NAME}"
  aws apprunner create-service --region "$AWS_REGION" --service-name "$APP_NAME" \
    --source-configuration '{
      "AuthenticationConfiguration": {"AccessRoleArn": "'"$ROLE_ARN"'"},
      "AutoDeploymentsEnabled": true,
      "ImageRepository": {
        "ImageIdentifier": "'"$IMAGE"'",
        "ImageRepositoryType": "ECR",
        "ImageConfiguration": {"Port": "8000"}
      }
    }' \
    --instance-configuration "{\"Cpu\": \"$CPU\", \"Memory\": \"$MEMORY\"}" \
    --health-check-configuration '{"Protocol": "HTTP", "Path": "/health"}' >/dev/null
else
  echo "==> Service exists, starting deployment of the new image"
  aws apprunner start-deployment --region "$AWS_REGION" --service-arn "$SERVICE_ARN" >/dev/null
fi

echo "==> Waiting for service to reach RUNNING"
for _ in $(seq 1 60); do
  read -r STATUS URL <<<"$(aws apprunner list-services --region "$AWS_REGION" \
    --query "ServiceSummaryList[?ServiceName=='${APP_NAME}'].[Status,ServiceUrl]" --output text)"
  if [ "$STATUS" = "RUNNING" ]; then
    echo "==> Deployed: https://${URL}   (interactive docs: https://${URL}/docs)"
    exit 0
  fi
  echo "    ${STATUS} ..."
  sleep 20
done
echo "!! service did not reach RUNNING in time — inspect with:"
echo "   aws apprunner describe-service --service-arn ${SERVICE_ARN:-<arn>} --region ${AWS_REGION}"
exit 1
