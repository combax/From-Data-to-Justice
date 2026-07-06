#!/usr/bin/env bash
# Remove everything deploy_aws.sh created: the App Runner service and the ECR
# repository. The shared AppRunnerECRAccessRole is left in place because other
# services may use it.
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
APP_NAME="${APP_NAME:-shr-offender-api}"

SERVICE_ARN=$(aws apprunner list-services --region "$AWS_REGION" \
  --query "ServiceSummaryList[?ServiceName=='${APP_NAME}'].ServiceArn" --output text)
if [ -n "$SERVICE_ARN" ]; then
  aws apprunner delete-service --service-arn "$SERVICE_ARN" --region "$AWS_REGION" >/dev/null
  echo "deleted App Runner service ${APP_NAME}"
else
  echo "no App Runner service named ${APP_NAME}"
fi

if aws ecr delete-repository --repository-name "$APP_NAME" --region "$AWS_REGION" --force >/dev/null 2>&1; then
  echo "deleted ECR repository ${APP_NAME}"
else
  echo "no ECR repository named ${APP_NAME}"
fi
