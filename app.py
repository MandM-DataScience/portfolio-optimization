from portfolio_optimization.aws_package import aws_fargate, lambda_handler

def handler(event, context):
    return lambda_handler(event, context)

if __name__ == '__main__':
    aws_fargate()