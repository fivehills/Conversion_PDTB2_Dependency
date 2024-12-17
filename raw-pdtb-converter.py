```python
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import re
from collections import defaultdict

@dataclass
class PDTBAnnotation:
    """Enhanced class for storing PDTB relation information"""
    type: str  # Explicit/Implicit/EntRel/AltLex/Hypophora
    conn_span: Optional[Tuple[int, int]]
    connective: str
    sense: List[str]  # Support multiple senses
    arg1_span: Tuple[int, int]
    arg2_span: Tuple[int, int]
    raw_line: str
    link: Optional[str] = None  # For LINK annotations
    source: Optional[str] = None  # For PDTB2/PDTB3 info
    supplementary_sense: Optional[List[str]] = None  # For additional senses
    edu_group: Optional[Tuple[int, int]] = None  # For local constituency structures

@dataclass
class Dependency:
    """Class for storing dependency information"""
    head: int
    dependent: int
    relation_type: str
    distance: int
    type: str
    connective: Optional[str] = None
    link: Optional[str] = None
    source: Optional[str] = None

class PDTBDependencyConverter:
    def __init__(self):
        # Complete list of asymmetric relations from Table 1
        self.asymmetric_relations = {
            "Contingency.Condition.Arg2-as-cond": ("Arg1", "Arg2"),
            "Contingency.Condition.Arg1-as-cond": ("Arg2", "Arg1"),
            "Contingency.negative-condition.Arg2-as-negcond": ("Arg1", "Arg2"),
            "Contingency.negative-condition.Arg1-as-negcond": ("Arg2", "Arg1"),
            "Contingency.Purpose.Arg2-as-goal": ("Arg1", "Arg2"),
            "Contingency.Purpose.Arg1-as-goal": ("Arg2", "Arg1"),
            "Comparison.Concession.Arg2-as-denier": ("Arg1", "Arg2"),
            "Comparison.Concession.Arg1-as-denier": ("Arg2", "Arg1"),
            "Comparison.Exception.Arg2-as-excpt": ("Arg1", "Arg2"),
            "Comparison.Exception.Arg1-as-excpt": ("Arg2", "Arg1"),
            "Comparison.Level-of-detail.Arg2-as-detail": ("Arg1", "Arg2"),
            "Comparison.Level-of-detail.Arg1-as-detail": ("Arg2", "Arg1"),
            "Comparison.Manner.Arg2-as-manner": ("Arg1", "Arg2"),
            "Comparison.Manner.Arg1-as-manner": ("Arg2", "Arg1"),
            "Comparison.Substitution.Arg2-as-subst": ("Arg1", "Arg2"),
            "Comparison.Substitution.Arg1-as-subst": ("Arg2", "Arg1")
        }

    def parse_span(self, span_text: str) -> Optional[Tuple[int, int]]:
        """Parse span text including discontinuous spans"""
        if not span_text or '..' not in span_text:
            return None
        try:
            if ';' in span_text:  # Handle discontinuous spans
                parts = span_text.split(';')
                start = int(parts[0].split('..')[0])
                end = int(parts[-1].split('..')[-1])
                return (start, end)
            else:
                start, end = map(int, span_text.split('..'))
                return (start, end)
        except (ValueError, IndexError):
            return None

    def span_to_edu(self, span: Tuple[int, int]) -> int:
        """Convert span to EDU ID"""
        return span[0] // 100 + 1 if span else 0

    def parse_pdtb_line(self, line: str) -> PDTBAnnotation:
        """Parse a single line of PDTB annotation"""
        parts = line.strip().split('|')
        
        # Basic fields
        rel_type = parts[0]
        
        # Parse connective span and text
        conn_span = self.parse_span(parts[1])
        connective = parts[7] if len(parts) > 7 and parts[7] else None
        
        # Get all senses
        senses = []
        supplementary_senses = []
        for field in parts:
            if any(x in field for x in ['Contingency', 'Temporal', 'Comparison', 'Expansion']):
                if not senses:
                    senses = field.split('.')
                else:
                    supplementary_senses.append(field.split('.'))
        
        # Find argument spans
        arg_spans = []
        for field in parts:
            span = self.parse_span(field)
            if span:
                arg_spans.append(span)
        
        # Get source and link information
        source = None
        link = None
        for field in parts:
            if 'PDTB' in field:
                source = field
            elif 'LINK' in field:
                link = field

        return PDTBAnnotation(
            type=rel_type,
            conn_span=conn_span,
            connective=connective,
            sense=senses,
            supplementary_sense=supplementary_senses,
            arg1_span=arg_spans[0] if arg_spans else (0, 0),
            arg2_span=arg_spans[1] if len(arg_spans) > 1 else (0, 0),
            raw_line=line,
            link=link,
            source=source
        )

    def determine_head_dependent(self, annotation: PDTBAnnotation) -> Tuple[int, int]:
        """Determine head and dependent EDUs based on relation type"""
        if not annotation.arg1_span or not annotation.arg2_span:
            return (0, 0)
            
        sense_str = '.'.join(annotation.sense)
        arg1_edu = self.span_to_edu(annotation.arg1_span)
        arg2_edu = self.span_to_edu(annotation.arg2_span)
        
        # Check asymmetric relations
        for pattern, (head_arg, dep_arg) in self.asymmetric_relations.items():
            if pattern in sense_str:
                if head_arg == "Arg1":
                    return (arg1_edu, arg2_edu)
                else:
                    return (arg2_edu, arg1_edu)
        
        # For symmetric relations, make Arg2 the head
        return (arg2_edu, arg1_edu)

    def identify_local_structures(self, annotations: List[PDTBAnnotation]) -> Dict[int, List[int]]:
        """Identify local constituency structures"""
        structures = defaultdict(list)
        
        for ann in annotations:
            arg1_edu = self.span_to_edu(ann.arg1_span)
            arg2_edu = self.span_to_edu(ann.arg2_span)
            
            if abs(arg1_edu - arg2_edu) == 1:
                min_edu = min(arg1_edu, arg2_edu)
                structures[min_edu].extend([arg1_edu, arg2_edu])
        
        return {k: sorted(list(set(v))) for k, v in structures.items()}

    def convert_to_dependencies(self, pdtb_text: str) -> List[Dependency]:
        """Convert PDTB annotations to dependencies"""
        # First parse all annotations
        annotations = [self.parse_pdtb_line(line) for line in pdtb_text.strip().split('\n')]
        
        # Identify local structures
        local_structures = self.identify_local_structures(annotations)
        
        # Convert to dependencies
        dependencies = []
        for annotation in annotations:
            # Skip certain relation types
            if annotation.type in ['EntRel', 'Hypophora']:
                continue
                
            # Determine head and dependent
            head, dependent = self.determine_head_dependent(annotation)
            if head == 0 or dependent == 0:
                continue
            
            # Create dependency
            dependency = Dependency(
                head=head,
                dependent=dependent,
                relation_type='.'.join(annotation.sense),
                distance=abs(head - dependent),
                type=annotation.type,
                connective=annotation.connective,
                link=annotation.link,
                source=annotation.source
            )
            dependencies.append(dependency)
        
        # Validate dependencies
        if not self.validate_dependencies(dependencies):
            raise ValueError("Invalid dependencies generated")
            
        return dependencies

    def validate_dependencies(self, dependencies: List[Dependency]) -> bool:
        """Validate converted dependencies"""
        for dep in dependencies:
            if dep.head <= 0 or dep.dependent <= 0:
                return False
            if not dep.relation_type:
                return False
            if dep.distance != abs(dep.head - dep.dependent):
                return False
        return True

    def calculate_discourse_distance(self, dependencies: List[Dependency]) -> float:
        """Calculate mean discourse distance"""
        if not dependencies:
            return 0.0
        
        total_distance = sum(dep.distance for dep in dependencies)
        return total_distance / len(dependencies)

def main():
    # Example usage with WSJ_0618 data
    sample_pdtb = """Explicit|96..100|||||9..78|when|Contingency.Condition.Arg2-as-cond||||||79..94||||||101..158|||||||||||96..100|PDTB2::wsj_0618::96..100::SAME|
Explicit|464..469||||||since|Temporal.Asynchronous.Succession||||||424..463||||||470..502|||||||||||464..469|PDTB2::wsj_0618::464..469::SAME|
Explicit|514..518||||||with|Contingency.Cause.Reason||||||509..513;579..612||||||519..577|||||||||ARGM-ADV|be|514..518|PDTB3|"""

    converter = PDTBDependencyConverter()
    
    try:
        # Convert to dependencies
        dependencies = converter.convert_to_dependencies(sample_pdtb)
        
        # Calculate discourse distance
        mean_distance = converter.calculate_discourse_distance(dependencies)
        
        # Print results
        print("\nConverted Dependencies:")
        for dep in dependencies:
            print(f"\nHead: {dep.head} → Dependent: {dep.dependent}")
            print(f"Relation: {dep.relation_type}")
            print(f"Type: {dep.type}")
            print(f"Distance: {dep.distance}")
            if dep.connective:
                print(f"Connective: {dep.connective}")
            if dep.link:
                print(f"Link: {dep.link}")
            if dep.source:
                print(f"Source: {dep.source}")
        
        print(f"\nMean Discourse Distance: {mean_distance:.2f}")
        
    except Exception as e:
        print(f"Error processing PDTB annotations: {str(e)}")

if __name__ == "__main__":
    main()
```
