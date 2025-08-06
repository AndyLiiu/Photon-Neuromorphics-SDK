# Terraform variables for Photon Neuromorphics SDK infrastructure

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  default     = "development"
  
  validation {
    condition = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be development, staging, or production."
  }
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "photon-neuro"
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "photon-neuro-cluster"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.27"
}

# Networking
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"  
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

# Database
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
  
  validation {
    condition = can(regex("^db\\.", var.db_instance_class))
    error_message = "DB instance class must start with 'db.'."
  }
}

# Redis
variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.micro"
  
  validation {
    condition = can(regex("^cache\\.", var.redis_node_type))
    error_message = "Redis node type must start with 'cache.'."
  }
}

# Logging
variable "log_retention_days" {
  description = "CloudWatch logs retention period"
  type        = number
  default     = 14
  
  validation {
    condition = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention days must be a valid CloudWatch retention value."
  }
}

# Domain
variable "domain_name" {
  description = "Domain name for the application (optional)"
  type        = string
  default     = ""
}

# Node groups configuration
variable "cpu_node_instance_types" {
  description = "Instance types for CPU compute nodes"
  type        = list(string)
  default     = ["c5.2xlarge", "c5.4xlarge"]
}

variable "gpu_node_instance_types" {
  description = "Instance types for GPU compute nodes"
  type        = list(string)
  default     = ["p3.2xlarge", "g4dn.2xlarge"]
}

variable "system_node_instance_types" {
  description = "Instance types for system nodes"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

# Auto-scaling
variable "cpu_nodes_min_size" {
  description = "Minimum number of CPU nodes"
  type        = number
  default     = 2
}

variable "cpu_nodes_max_size" {
  description = "Maximum number of CPU nodes"
  type        = number
  default     = 10
}

variable "cpu_nodes_desired_size" {
  description = "Desired number of CPU nodes"
  type        = number
  default     = 3
}

variable "gpu_nodes_min_size" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 0
}

variable "gpu_nodes_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 5
}

variable "gpu_nodes_desired_size" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 1
}

# Security
variable "enable_cluster_encryption" {
  description = "Enable EKS cluster encryption"
  type        = bool
  default     = true
}

variable "enable_irsa" {
  description = "Enable IAM Roles for Service Accounts"
  type        = bool
  default     = true
}

# Monitoring and observability
variable "enable_cloudwatch_logging" {
  description = "Enable CloudWatch container insights"
  type        = bool
  default     = true
}

variable "enable_prometheus" {
  description = "Enable Prometheus monitoring"
  type        = bool
  default     = true
}

variable "enable_grafana" {
  description = "Enable Grafana dashboards"
  type        = bool
  default     = true
}

# Storage
variable "ebs_volume_size" {
  description = "EBS volume size for nodes (GB)"
  type        = number
  default     = 100
  
  validation {
    condition = var.ebs_volume_size >= 20 && var.ebs_volume_size <= 16000
    error_message = "EBS volume size must be between 20 and 16000 GB."
  }
}

variable "enable_ebs_encryption" {
  description = "Enable EBS volume encryption"
  type        = bool
  default     = true
}

# Backup and disaster recovery
variable "backup_retention_period" {
  description = "Database backup retention period (days)"
  type        = number
  default     = 7
  
  validation {
    condition = var.backup_retention_period >= 0 && var.backup_retention_period <= 35
    error_message = "Backup retention period must be between 0 and 35 days."
  }
}

variable "enable_multi_az" {
  description = "Enable multi-AZ deployment for RDS"
  type        = bool
  default     = false
}

# Cost optimization
variable "use_spot_instances" {
  description = "Use spot instances for worker nodes"
  type        = bool
  default     = true
}

variable "spot_instance_pools" {
  description = "Number of spot instance pools to use"
  type        = number
  default     = 3
  
  validation {
    condition = var.spot_instance_pools >= 1 && var.spot_instance_pools <= 10
    error_message = "Spot instance pools must be between 1 and 10."
  }
}

# Performance tuning
variable "enable_accelerated_networking" {
  description = "Enable enhanced networking for instances"
  type        = bool
  default     = true
}

variable "enable_placement_groups" {
  description = "Enable placement groups for GPU nodes"
  type        = bool
  default     = false
}

# Compliance and governance
variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest for all services"
  type        = bool
  default     = true
}

variable "enable_encryption_in_transit" {
  description = "Enable encryption in transit for all services"
  type        = bool
  default     = true
}

variable "compliance_framework" {
  description = "Compliance framework to follow (none, hipaa, pci, sox)"
  type        = string
  default     = "none"
  
  validation {
    condition = contains(["none", "hipaa", "pci", "sox"], var.compliance_framework)
    error_message = "Compliance framework must be none, hipaa, pci, or sox."
  }
}

# Tags
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = ""
}

variable "project_owner" {
  description = "Project owner email"
  type        = string
  default     = ""
}

# Local values for computed configurations
locals {
  # Common tags
  common_tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
      CreatedAt   = timestamp()
    },
    var.cost_center != "" ? { CostCenter = var.cost_center } : {},
    var.project_owner != "" ? { Owner = var.project_owner } : {},
    var.additional_tags
  )
  
  # Environment-specific configurations
  env_config = {
    development = {
      db_instance_class     = "db.t3.micro"
      redis_node_type      = "cache.t3.micro"
      backup_retention     = 1
      multi_az            = false
      deletion_protection = false
    }
    staging = {
      db_instance_class     = "db.t3.small"
      redis_node_type      = "cache.t3.small"
      backup_retention     = 3
      multi_az            = false
      deletion_protection = false
    }
    production = {
      db_instance_class     = "db.r5.large"
      redis_node_type      = "cache.r5.large"
      backup_retention     = 30
      multi_az            = true
      deletion_protection = true
    }
  }
  
  # Current environment configuration
  current_env = local.env_config[var.environment]
  
  # Cluster name with environment suffix
  full_cluster_name = "${var.cluster_name}-${var.environment}"
}