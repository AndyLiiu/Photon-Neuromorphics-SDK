# Terraform configuration for Photon Neuromorphics SDK infrastructure
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "PhotonNeuromorphicsSDK"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project_name}-vpc-${var.environment}"
  cidr = var.vpc_cidr

  azs             = data.aws_availability_zones.available.names
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true

  tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }

  public_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb" = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb" = "1"
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = var.kubernetes_version

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true
  cluster_endpoint_private_access = true

  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  # Node groups
  eks_managed_node_groups = {
    # CPU compute nodes
    cpu_nodes = {
      name = "photon-neuro-cpu"
      
      instance_types = ["c5.2xlarge", "c5.4xlarge"]
      capacity_type  = "SPOT"
      
      min_size     = 2
      max_size     = 10
      desired_size = 3

      labels = {
        node-type = "compute"
      }

      taints = [
        {
          key    = "node-type"
          value  = "compute"
          effect = "NO_SCHEDULE"
        }
      ]

      ami_type = "AL2_x86_64"
      disk_size = 100
      
      vpc_security_group_ids = [aws_security_group.node_group_sg.id]
    }

    # GPU compute nodes
    gpu_nodes = {
      name = "photon-neuro-gpu"
      
      instance_types = ["p3.2xlarge", "p3.8xlarge", "g4dn.2xlarge"]
      capacity_type  = "SPOT"
      
      min_size     = 0
      max_size     = 5
      desired_size = 1

      labels = {
        node-type = "gpu-compute"
        accelerator = "nvidia-tesla-v100"
      }

      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]

      ami_type = "AL2_x86_64_GPU"
      disk_size = 200
      
      vpc_security_group_ids = [aws_security_group.node_group_sg.id]
    }

    # System/management nodes
    system_nodes = {
      name = "photon-neuro-system"
      
      instance_types = ["t3.medium", "t3.large"]
      capacity_type  = "ON_DEMAND"
      
      min_size     = 1
      max_size     = 3
      desired_size = 2

      labels = {
        node-type = "system"
      }

      vpc_security_group_ids = [aws_security_group.node_group_sg.id]
    }
  }

  tags = {
    Environment = var.environment
    Application = "PhotonNeuromorphicsSDK"
  }
}

# Security Groups
resource "aws_security_group" "node_group_sg" {
  name_prefix = "${var.cluster_name}-node-group-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-node-group-sg"
  }
}

# RDS for metadata storage
resource "aws_db_subnet_group" "photon_neuro" {
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Name = "${var.project_name}-db-subnet-group"
  }
}

resource "aws_security_group" "rds" {
  name_prefix = "${var.project_name}-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "PostgreSQL"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  tags = {
    Name = "${var.project_name}-rds-sg"
  }
}

resource "aws_db_instance" "photon_neuro" {
  identifier = "${var.project_name}-db-${var.environment}"

  engine         = "postgres"
  engine_version = "15"
  instance_class = var.db_instance_class
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_encrypted     = true

  db_name  = "photon_neuro"
  username = "photon_user"
  password = random_password.db_password.result

  db_subnet_group_name   = aws_db_subnet_group.photon_neuro.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = var.environment == "development"
  deletion_protection = var.environment == "production"

  tags = {
    Name = "${var.project_name}-db"
  }
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

# ElastiCache Redis for caching
resource "aws_elasticache_subnet_group" "photon_neuro" {
  name       = "${var.project_name}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "elasticache" {
  name_prefix = "${var.project_name}-elasticache-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "Redis"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  tags = {
    Name = "${var.project_name}-elasticache-sg"
  }
}

resource "aws_elasticache_replication_group" "photon_neuro" {
  replication_group_id       = "${var.project_name}-redis-${var.environment}"
  description                = "Redis cluster for Photon Neuromorphics SDK"

  node_type            = var.redis_node_type
  port                 = 6379
  parameter_group_name = "default.redis7"

  num_cache_clusters = 2
  
  subnet_group_name  = aws_elasticache_subnet_group.photon_neuro.name
  security_group_ids = [aws_security_group.elasticache.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  tags = {
    Name = "${var.project_name}-redis"
  }
}

# S3 for model and data storage
resource "aws_s3_bucket" "photon_neuro_models" {
  bucket = "${var.project_name}-models-${var.environment}-${random_id.bucket_suffix.hex}"
}

resource "aws_s3_bucket" "photon_neuro_data" {
  bucket = "${var.project_name}-data-${var.environment}-${random_id.bucket_suffix.hex}"
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# S3 bucket configurations
resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.photon_neuro_models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "models" {
  bucket = aws_s3_bucket.photon_neuro_models.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.photon_neuro_models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM roles and policies
resource "aws_iam_role" "photon_neuro_app" {
  name = "${var.project_name}-app-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${replace(module.eks.oidc_provider_arn, "/^.*oidc-provider//", "")}:sub": "system:serviceaccount:photon-neuro:photon-neuro-sa"
            "${replace(module.eks.oidc_provider_arn, "/^.*oidc-provider//", "")}:aud": "sts.amazonaws.com"
          }
        }
      }
    ]
  })
}

resource "aws_iam_policy" "photon_neuro_app" {
  name = "${var.project_name}-app-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.photon_neuro_models.arn,
          "${aws_s3_bucket.photon_neuro_models.arn}/*",
          aws_s3_bucket.photon_neuro_data.arn,
          "${aws_s3_bucket.photon_neuro_data.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.db_credentials.arn
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "photon_neuro_app" {
  role       = aws_iam_role.photon_neuro_app.name
  policy_arn = aws_iam_policy.photon_neuro_app.arn
}

# Secrets Manager for sensitive data
resource "aws_secretsmanager_secret" "db_credentials" {
  name = "${var.project_name}/db-credentials/${var.environment}"
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = aws_db_instance.photon_neuro.username
    password = aws_db_instance.photon_neuro.password
    endpoint = aws_db_instance.photon_neuro.endpoint
    port     = aws_db_instance.photon_neuro.port
    dbname   = aws_db_instance.photon_neuro.db_name
  })
}

# CloudWatch for logging and monitoring
resource "aws_cloudwatch_log_group" "photon_neuro" {
  name              = "/aws/eks/${var.cluster_name}/photon-neuro"
  retention_in_days = var.log_retention_days

  tags = {
    Application = "PhotonNeuromorphicsSDK"
    Environment = var.environment
  }
}

# Route53 for DNS (if domain is provided)
resource "aws_route53_zone" "main" {
  count = var.domain_name != "" ? 1 : 0
  name  = var.domain_name

  tags = {
    Name = "${var.project_name}-zone"
  }
}

resource "aws_acm_certificate" "main" {
  count           = var.domain_name != "" ? 1 : 0
  domain_name     = var.domain_name
  validation_method = "DNS"

  subject_alternative_names = [
    "*.${var.domain_name}"
  ]

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "${var.project_name}-cert"
  }
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.photon_neuro.endpoint
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.photon_neuro.primary_endpoint_address
}

output "models_bucket" {
  description = "S3 bucket for models"
  value       = aws_s3_bucket.photon_neuro_models.bucket
}

output "data_bucket" {
  description = "S3 bucket for data"
  value       = aws_s3_bucket.photon_neuro_data.bucket
}